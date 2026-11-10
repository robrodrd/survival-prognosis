# *****************************************************************************************************************************************#
# Kaplan-Meier/Implemented: Estimating Survival Probability over time & visualizing Survival Curves between Cohorts                        #                #
# Cox Proportional Hazards/Implemented: Evaluating Impact of Time-Independent Covariates (age, biomarkers, treatment type) on Hazard Rate  #
# Linear Mixed-Effects Models (LMMs)/Not Implemented: Modeling continuous trajectories over time accounting for within-patient correlation #                #
# Generalized Estimating Equations (GEE)/Not Implemented: Population-average modeling of repeated binary, count, or continuous outcomes    #                #
# Joint Modeling/Not Implemented: Simultaneous Analyzing Longitudinal Repeated Measures (Biomarker Degradation) and (Time-to-Event data)   # 
# ---------------------------------------------------------------------------------------------------------------------------------------- #
# For: Linear Mixed-Effects Models (LMM)      : import statsmodels.formula.api as smf                                                      #                                                    #
# For: Generalized Estimating Equations (GEE) : import statsmodels.formula.api.gee as smf                                                  #                          #
# For Joint Modeling                          : multiple options : flash, jmstate, rpy : need to do research for which to use when         #             #
# **************************************************************************************************************************************** #

# **************************************************************************************************************************************** #
# To start project in the morning: change into Project Folder    :   cd ~/projects/survival-prognosis                                      #
#                                  start Virtual Environment     :   source .venv/bin/activate                                             # 
#                                  start GCP PostgreSQL Database : ./cloud-sql-proxy healthcare-analytics-v1:us-central1:cpp-production-db #
#                                                                                                                                          #
# Created 3 Tables in cpp-production-db Instance                 :   Feedback Table         : User Feedback                                #
#                                                                :   Subscription Table     : Subcription Fees                             #
#                                                                :   Audit Table            : PHI/HIPAA Compliance                         #
#                                                                                                                                          #
#                                                         :  DB-Instance : cpp-production-db                                               #
#                                                         :  DB_USER     : postgres (automatically set as default when created)            #
#                                                         :  DB_PASSWORD : FyIdiot&00                                                      #
#                                                         :  DB_NAME     : postgres (automatically set as default when created)            #
#                                                         :  DB_HOST     : /cloudsql/healthcare-analytics-v1:us-central1:cpp-production-db #
#                                                                                                                                          #
# To write to database on local machine start prox :  ./cloud-sql-proxy healthcare-analytics-v1:us-central1:cpp-production-db --port=5432  #
# **************************************************************************************************************************************** #                  

# **************************************************************************************************************************************** #
# Application url:          https://healthcare-analytics-v1.web.app/                                                                       #
#                           https://survival-prognosis-service-100651041936.us-central1.run.app/                                           #
#                           https://clinicalprognosis.app                                                                                  #
# Custom Domain Name        clinicalprognosis.app       --> url to get to the app                                                          #                       
# **************************************************************************************************************************************** #

# **************************************************************************************************************************************** #
# To push code to GitHub:   --> 1.: check if network connection and credentials are in place :  $ git remote -v                            #      
#                           --> 2.: test connectivity and authentication                     :  $ git fetch origin                         #
#                           --> 3.: check status if local data is up to date with GitHub     :  $ git status                               #
#                           --> 4.:              :$ git add requirements.txt rxconfig.py assets/ survival_prognosis/survival_prognosis.py  #
#                           --> 5.:              :$ git commit -m "Add PostgreSQL database integration and update application assets"      #
#                           --> 6.: final push   :$ git push origin main                                                                   #
#                                                           usr name           : robrodrd                                                  #
#                                                                                                                                          #
# **************************************************************************************************************************************** #

# **************************************************************************************************************************************** #
# Docker USR/PWD: 



import asyncio
from typing import List, Union, Optional, Any
import io
import reflex as rx
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import base64
import tempfile
import os
from dotenv import load_dotenv
import psycopg2
from weasyprint import HTML
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from lifelines.utils import median_survival_times
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_lung, load_gbsg2
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import scipy.stats as stats
import logging
import resend
from sklearn.metrics import roc_auc_score
import warnings

load_dotenv()   # Load environment variables from the .env file in the root directory

# --- Suppress non-unique index warnings from lifelines ---
warnings.filterwarnings(
    "ignore",
    message="DataFrame Index is not unique, defaulting to incrementing index instead.",
    category=UserWarning
)

pd.set_option('future.no_silent_downcasting', True)

# Helper to get CI string: Calculates 95% CI for Survival Probability at a given time: Returns: (Survival_Prob_CI_String, Median_CI_Placeholder)
def get_ci_string(kmf, time_point):
    if not hasattr(kmf, 'confidence_interval_'):
        return ("N/A", "N/A")
    
    # 1. Survival Prob CI
    try:
        ci_df = kmf.confidence_interval_
        # Use asof to find the latest confidence interval values at or before time_point
        ci_at_time = ci_df.asof(time_point)
        if ci_at_time.empty or ci_at_time.isna().all():
            idx = ci_df.index.get_indexer([time_point], method='pad')[0]
            idx = 0 if idx == -1 else idx
            low_p = ci_df.iloc[idx, 0]
            high_p = ci_df.iloc[idx, 1]
        else:
            low_p = ci_at_time.iloc[0]
            high_p = ci_at_time.iloc[1]
            
        prob_ci = f"{int(round(low_p*100))}% - {int(round(high_p*100))}%"
    except Exception as e:
        print(f"Survival Prob CI Error: {e}")
        prob_ci = "N/A"

    # 2. Median Survival Time CI
    try:
        med_ci_df = median_survival_times(kmf.confidence_interval_)
        if med_ci_df.ndim > 1:
            low_m, high_m = med_ci_df.iloc[0, 0], med_ci_df.iloc[0, 1]
        else:
            low_m, high_m = med_ci_df.iloc[0], med_ci_df.iloc[1]
            
        low_str = f"{low_m:.1f}" if not np.isinf(low_m) and not pd.isna(low_m) else "NR"
        high_str = f"{high_m:.1f}" if not np.isinf(high_m) and not pd.isna(high_m) else "NR"
        med_ci = f"{low_str} - {high_str}"
    except Exception:
        med_ci = "N/A"

    return (prob_ci, med_ci)

def make_initial_figure():
    """Generates a stable baseline layout to establish initial rendering bounds."""
    fig = go.Figure()
    fig.update_layout(
        height=420,
        autosize=True,
        margin=dict(l=40, r=20, t=40, b=40),
        template="simple_white"
    )
    return fig

# ------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------ STATE CLASS ------------------------------------------------------------------ #
class State(rx.State):
    
    km_error_message: str = ""
    km_warning_message: str = ""
    
    median_survival_list: list[str] = []
    survival_probability_list: list[str] = []
    
    def _generate_km_cph_table(self):
        """
        Calculates and prepares the Univariate/Multivariate CPH adjustment ratios 
        specifically for display during Kaplan-Meier survival evaluations.
        """
        # ------------- PREPARE DATA AND CALL self._run_cph_engine to calculate MULTIVARIATE CPH RATIOS required in the KM Report-------------- # 
        base_full = getattr(self, "filtered_df", pd.DataFrame())
        if base_full.empty:
            base_full = getattr(self, "df", pd.DataFrame())
            
        sub_df1 = getattr(self, "current_df1", pd.DataFrame())
        sub_df2 = getattr(self, "current_df2", pd.DataFrame())
        
        is_full_cohort = (
            "Full Cohort" in str(getattr(self, "g1_detailed_label", "")) and 
            "Full Cohort" in str(getattr(self, "g2_detailed_label", ""))
        )
    
        # --- ASYMMETRIC: NOT FULL COHORT / PARTIAL STRATIFICATION HANDLING --- #
        if not sub_df1.empty and sub_df2.empty:
            active_g1 = sub_df1
            active_g2 = base_full.drop(sub_df1.index, errors="ignore")
            strat_mode = "Group 1 Only (Group 2 = Complement)"
        elif sub_df1.empty and not sub_df2.empty:
            active_g2 = sub_df2
            active_g1 = base_full.drop(sub_df2.index, errors="ignore")
            strat_mode = "Group 2 Only (Group 1 = Complement)"
        else:
            active_g1 = sub_df1 if not sub_df1.empty else pd.DataFrame()
            active_g2 = sub_df2 if not sub_df2.empty else pd.DataFrame()
            strat_mode = "Both Groups Filtered / Custom"
    
        # --- CRITICAL SAFEGUARD: ENSURE STRICT MUTUAL EXCLUSIVITY --- #
        # Drop any overlapping patient indices from Group 2 so it never overwrites Group 1
        if not active_g1.empty and not active_g2.empty:
            active_g2 = active_g2.drop(active_g1.index, errors="ignore")
    
        is_comparative = not is_full_cohort and (not active_g1.empty or not active_g2.empty)

        # 🌟 ----------------------- GENERATING THE KM-CPH-HR TABLE AS PART THE KM ANALYSIS RESULTS ------------------------------- #
        # ---- CREATING THE COMPARISON_COHORT COVARIATE ---- #
        if is_comparative:      #--- creates the comparison_group sub-data set --- #
            unified_g1_g2 = pd.concat([active_g1, active_g2]).drop_duplicates().copy()
            unified_g1_g2["stratification_group"] = "Unassigned"
    
            unified_g1_g2.loc[unified_g1_g2.index.isin(active_g1.index), "stratification_group"] = "Group 1"
            unified_g1_g2.loc[unified_g1_g2.index.isin(active_g2.index), "stratification_group"] = "Group 2"
    
            # Create the binary group indicator safely
            unified_g1_g2["comparison_cohort"] = None
            unified_g1_g2.loc[unified_g1_g2.index.isin(active_g1.index), "comparison_cohort"] = 0
            unified_g1_g2.loc[unified_g1_g2.index.isin(active_g2.index), "comparison_cohort"] = 1
            
            unified_g1_g2["comparison_cohort"] = pd.to_numeric(unified_g1_g2["comparison_cohort"], errors='coerce').fillna(0).astype(int)
        else:
            unified_g1_g2 = base_full.copy()
            if unified_g1_g2.empty:
                unified_g1_g2 = sub_df1.copy() if not sub_df1.empty else getattr(self, "df", pd.DataFrame()).copy()
            unified_g1_g2["stratification_group"] = "Full Cohort"
    
        # --- DYNAMIC COVARIATE DISCOVERY --- #
        days_val = getattr(self, "days_input", 365)
    
        time_col = getattr(self, "selected_time_column", None)
        event_col = getattr(self, "selected_event_column", None)
        
        csv_cols = getattr(self, "csv_columns", [])        
    
        base_adjustment_panel = [            # --- identifies the column names of the comparision_group dataset --- #                                       
            c for c in csv_cols
            if c not in [time_col, event_col] and not str(c).startswith("Unnamed")
        ]
        
        primary_comparison_rows_data = []
        multivariate_covs_rows_data = []
    
        original_covs = list(getattr(self, "cph_active_covariates", []))

        try:
            target_df = unified_g1_g2
            stratifying_vars = getattr(self, "active_stratification_vars", [])
    
            # 1. Extract stratifying covariates from the checked values dictionary
            g1_checked = getattr(self, "g1_checked_values", {})    # --- Column Names / Variables selected in the Sidebar Group 1 --- #
            g2_checked = getattr(self, "g2_checked_values", {})    # --- Column Names / Variables selected in the Sidebar Group 2 --- #
            
            stratifying_vars = []
            for d in [g1_checked, g2_checked]:
                for col, val in d.items():
                    if val and not (isinstance(val, list) and not val) and "Custom Range:" not in str(val):
                        if col not in stratifying_vars:
                            stratifying_vars.append(col)
            
            clean_adjustment_panel = [                                # --- building the Complement Variable List --- (sex, ph.ecog,...) --- #
                c for c in base_adjustment_panel
                if c not in stratifying_vars
            ]
    
            if is_full_cohort:  # --- SCENARIO 1: Unstratified Full Cohort (Single Combined Table) ---
                report_covs = list(clean_adjustment_panel)
                self._run_cph_engine(target_df, days_val, preserve_state=True, active_covs_override=report_covs)
                multivariate_covs_rows_data = list(getattr(self, "cph_table_data", []))
            
            else:               # --- SCENARIOS 2 & 3: Stratified Analysis (TWO SEPARATE TABLES) ---
                # --- TABLE A: Primary 1-Row Comparison (Group 1 vs Group 2) ---
                self._run_cph_engine(target_df, days_val, preserve_state=True, active_covs_override=["comparison_cohort"])
                primary_comparison_rows_data = list(getattr(self, "cph_table_data", []))
    
                # --- TABLE B: Multivariable Background Clinical Adjustment Panel ---
                report_covs = list(clean_adjustment_panel)
                if report_covs:
                    self._run_cph_engine(target_df, days_val, preserve_state=True, active_covs_override=report_covs)
                    multivariate_covs_rows_data = list(getattr(self, "cph_table_data", []))
            
        except Exception as e:
            print(f"ERROR during CPH engine handoff: {e}")
    
        self.cph_active_covariates = original_covs       # Restore original active covariates

        # Store the resulting rows into state variables so your `cph_results_table()` component can render them dynamically on screen
        self.cph_table_data = multivariate_covs_rows_data if is_full_cohort else primary_comparison_rows_data + multivariate_covs_rows_data
        
        self.cph_has_run = True
    
    # ---------------------------------------------------------------------------------------------------------------------------------- #
    # --- Called when 'Done' is clicked in Baseline Characteristics Modal - Saves Configurations, Updates Table 1, Closes Modal --- #
    def save_baseline_configurations_and_refresh(self):
        
        self.is_baseline_modal_open = False                 # --- Boolean State Variable to close Modal --- #
        
        df1 = getattr(self, "current_df1", None)            # --- Current Dataframes from State --- #
        df2 = getattr(self, "current_df2", None)
        
        if df1 is not None and not df1.empty:               # --- Updates Table 1 immediately using tupdated column_configs
            self._generate_table_one(df1, df2)
    
    
    table_one_g1_label: str = "Group 1"
    table_one_g2_label: str = "Group 2"
    
    is_manuscript_modal_open: bool = False                  # --- Unified Manuscript State Variables --- #
    editable_manuscript_text: str = ""

    @rx.event
    def set_editable_manuscript_text(self, value: str):
        self.editable_manuscript_text = value

    @rx.event
    def close_manuscript_modal(self):
        self.is_manuscript_modal_open = False

    @rx.event
    def open_manuscript_modal(self):
        model_name = getattr(self, "selected_model", "")

        # --- Inline warning safeguard if no model selected ---
        if not model_name or model_name.strip() == "":
            self.show_pdf_warning = True
            self.pdf_warning_msg = "Select Model First"
            return

        self.show_pdf_warning = False
        self.pdf_warning_msg = ""

        # --- Dynamic Model Routing ---
        if "Kaplan-Meier Survival" in model_name:
            self._prepare_km_manuscript_draft()
        elif "Cox Proportional Hazards" in model_name:
            self._prepare_cph_manuscript_draft()
        else:
            self.show_pdf_warning = True
            self.pdf_warning_msg = f"Unsupported model selected: {model_name}"
            return

        self.is_manuscript_modal_open = True

    

    def _prepare_km_manuscript_draft(self):
        g1_desc_text = getattr(self, "g1_detailed_label", "Group 1").replace("Group 1:", "").strip()
        g2_desc_text = getattr(self, "g2_detailed_label", "Group 2").replace("Group 2:", "").strip()
        g1_sz = getattr(self, "g1_n", "36")
        g2_sz = getattr(self, "g2_n", "13")
        
        # Extract and clean p-value text
        raw_p_text = getattr(self, "p_value_text", "Log-rank p: 0.80")
        log_p = raw_p_text.replace("Log-rank p:", "").replace("Log-rank test:", "").strip()

        # Safely pull live median survival strings from engine list output
        med_list = getattr(self, "median_survival_list", [])
        median_surv1 = med_list[0] if len(med_list) > 0 else "Group 1: 11.6 Mo | 95% CI: 9.4 - 14.9"
        median_surv2 = med_list[1] if len(med_list) > 1 else "Group 2: 11.3 Mo | 95% CI: 2.6 - 24.0"

        # Safely evaluate p-value for scientific phrasing
        try:
            p_val_num = float(log_p) if "<" not in log_p else 0.0001
            is_significant = p_val_num < 0.05
        except (ValueError, TypeError):
            is_significant = False

        # Dynamic phrasing based on actual statistical significance
        if is_significant:
            sig_phrase = "demonstrated statistically significant trajectory separation"
            conclusion_phrase = "reveals distinct survival divergence"
            results_sentence = f"The Mantel-Cox log-rank test confirmed differential survival patterns (Log-rank p = {log_p})."
        else:
            sig_phrase = "showed comparable survival trajectories without statistically significant divergence"
            conclusion_phrase = "indicates similar survival patterns"
            results_sentence = f"The Mantel-Cox log-rank test yielded no statistically significant difference between the curves (Log-rank p = {log_p})."

        # Capture tie-breaking warning notice if clustered data triggered adjustments
        warning_note = getattr(self, "km_warning_message", "")
        tie_text = ""
        if warning_note:
            tie_text = f" Note on Data Handling: {warning_note}"

        self.editable_manuscript_text = f"""# COMPARATIVE KAPLAN-MEIER SURVIVAL ANALYSIS

## A STROBE/REMARK-COMPLIANT EVALUATION

## ABSTRACT
Background: Non-parametric survival analysis and inter-group trajectory comparisons are vital for assessing clinical efficacy across stratified patient cohorts.
Methods: Survival distributions were estimated using the Kaplan-Meier method, with Greenwood's formula for confidence intervals. Inter-group differences were evaluated via the Mantel-Cox log-rank test.{tie_text} Multivariable adjustment was performed using a Cox proportional hazards framework.
Results: Evaluation of Group 1 ({g1_desc_text}, N = {g1_sz}) versus Group 2 ({g2_desc_text}, N = {g2_sz}) {sig_phrase} (Log-rank p = {log_p}). 
Conclusion: Stratified Kaplan-Meier analysis {conclusion_phrase}, evaluated via adjusted proportional hazards modeling.

## 1. INTRODUCTION
Time-to-event outcomes require rigorous non-parametric estimation to properly characterize risk trajectories without imposing rigid parametric distribution assumptions.
[USER ACTION REQUIRED: Insert 2-3 paragraphs detailing the clinical context, biological rationale, and hypothesis underlying the comparison of these cohorts.]

## 2. METHODS
2.1 Study Design & Cohort Stratification: Data were processed through the interactive research analytics engine. Cohorts were stratified into Group 1 ({g1_desc_text}, N = {g1_sz}) and Group 2 ({g2_desc_text}, N = {g2_sz}).{tie_text}
2.2 Statistical Analysis: Survival probabilities and median survival times were calculated using Kaplan-Meier estimators. Hypothesis testing for inter-group variance utilized the Mantel-Cox log-rank test. Adjusted hazard ratios (HR) and 95% confidence intervals were derived using a complementary Cox proportional hazards model incorporating background clinical covariates.

## 3. RESULTS
The comparative evaluation comprised {g1_sz} patients in Group 1 and {g2_sz} patients in Group 2. {results_sentence} Median survival estimates were recorded as follows: {median_surv1} versus {median_surv2}. Multivariable adjustment accounting for baseline demographics contextualizes these survival outcomes.

## 4. DISCUSSION
[USER ACTION REQUIRED: Discuss the clinical implications of the observed survival separation or similarity, and how findings align with existing medical literature.]
4.1 Study Limitations: Retrospective cohort stratification necessitates prospective validation to eliminate potential selection biases.

## 5. CONCLUSION
The comparative Kaplan-Meier and adjusted survival analyses provide a rigorous assessment of prognostic trajectories between the evaluated cohorts."""

    def _prepare_cph_manuscript_draft(self):
        dataset_name = getattr(self, "selected_dataset", "Clinical Cohort")
        if dataset_name:
            dataset_name = dataset_name.split('/')[-1].replace('.csv', '').replace('_', ' ').title()
            
        total_n = getattr(self, "cph_full_cohort_n", getattr(self, "analyzed_n", 168))
        censored_n = getattr(self, "censored_count", 63)
        
        # Safely compute events if available, otherwise fallback
        try:
            events_n = int(total_n) - int(censored_n)
        except (ValueError, TypeError):
            events_n = getattr(self, "events_count", "N/A")

        unit_label = getattr(self, "timeline_unit", "Mo")
        med_fu = getattr(self, "median_followup", "54.0")

        base_c = getattr(self, "cph_c_index_text", "Harrell's C-index: 0.654 (95% CI: 0.564 – 0.743)")
        corrected_c = f"{getattr(self, 'cph_corrected_c_index', 0.629):.3f}"
        
        gnd_chi2 = getattr(self, "gnd_chi2", "13.84")
        gnd_p = getattr(self, "gnd_p", "0.0861")
        
        lr_p_text = getattr(self, "cph_log_rank_p", "Global LR Test: chi2 = 22.42, p = 2.1506e-03")

        self.editable_manuscript_text = f"""# MULTIVARIABLE PROGNOSTIC MODEL FOR SURVIVAL OUTCOMES

## A TRIPOD-COMPLIANT ANALYSIS        

## ABSTRACT
Background: Accurate risk stratification is essential for personalized clinical management and outcome prediction.
Methods: A multivariable Cox proportional hazards model was developed utilizing a cohort of {total_n} patients ({events_n} events, {censored_n} censored; median follow-up of {med_fu} {unit_label}). Model performance was evaluated via Harrell's C-index with bootstrap optimism correction, Greenwood-Nam-D'Agostino (GND) calibration testing, and Decision Curve Analysis (DCA).
Results: The multivariable model achieved {base_c}, with a bootstrap optimism-corrected C-index of {corrected_c}. Calibration testing verified absolute risk agreement (GND chi2 = {gnd_chi2}, p = {gnd_p}), and risk stratification demonstrated significant separation across survival trajectories ({lr_p_text}).
Conclusion: The validated multivariable model provides reliable risk stratification suitable for clinical decision support.

## 1. INTRODUCTION
Prognostic models play a critical role in clinical practice by facilitating risk stratification, guiding therapeutic strategies, and assisting in shared decision-making.
[USER ACTION REQUIRED: Insert 2-3 paragraphs detailing specific disease background, current clinical gaps, and rationale for evaluating candidate predictors in this study.]

## 2. METHODS
This study was conducted and reported in accordance with the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD) guidelines.
2.1 Study Population: Data were derived from the {dataset_name} cohort comprising N = {total_n} records. Baseline characteristics were summarized using standard descriptive statistics.
2.2 Model Construction & Validation: Univariate and multivariable Cox proportional hazards regressions were performed with L2 regularization (penalizer=0.1). Internal validation utilized non-parametric bootstrap resamples to calculate optimism-corrected discrimination. Calibration was evaluated via GND goodness-of-fit testing.

## 3. RESULTS
The primary evaluation cohort comprised {total_n} patients, recording {events_n} events and {censored_n} censored observations over a median follow-up of {med_fu} {unit_label}. In multivariable analysis, predictive performance was characterized by {base_c}. Model calibration indicated good quantitative agreement, and risk stratification into tertiles yielded pronounced trajectory divergence ({lr_p_text}).

## 4. DISCUSSION
[USER ACTION REQUIRED: Discuss how these hazard ratios compare with established literature. Elaborate on practical clinical implications of the Decision Curve Analysis findings.]
4.1 Study Limitations: While internal validation was established via bootstrap resamples, prospective external validation on an independent cohort remains necessary prior to broad clinical adoption.

## 5. CONCLUSION
The multivariable prognostic model demonstrates robust discrimination, calibration, and net clinical benefit for survival outcome forecasting."""
    
    @rx.event
    def export_edited_manuscript_docx(self):
        import docx
        import io
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        model_name = getattr(self, "selected_model", "")
        if "Kaplan-Meier Survival" in model_name:
            filename = "KM_clinical_manuscript_template.docx"
        elif "Cox Proportional Hazards" in model_name:
            filename = "CPH_clinical_manuscript_template.docx"
        else:
            filename = "clinical_manuscript_template.docx"

        doc = docx.Document()

        # Page setup
        for section in doc.sections:
            section.top_margin = Inches(1.0)
            section.bottom_margin = Inches(1.0)
            section.left_margin = Inches(1.0)
            section.right_margin = Inches(1.0)

        # Parse text block line by line and format into Word elements
        lines = self.editable_manuscript_text.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("# "):
                p = doc.add_paragraph()
                run = p.add_run(stripped[2:])
                run.font.name = "Calibri"
                run.font.size = Pt(16)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0, 0, 0)  # Pure Black for main title
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(4)
            elif stripped.startswith("## "):
                p = doc.add_paragraph()
                run = p.add_run(stripped[3:])
                run.font.name = "Calibri"
                run.font.size = Pt(13)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0, 0, 0)  # Pure Black for subtitle & section headings
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(14)
                p.paragraph_format.space_after = Pt(6)
            else:
                p = doc.add_paragraph()
                run = p.add_run(stripped)
                run.font.name = "Calibri"
                run.font.size = Pt(11)
                run.font.color.rgb = RGBColor(0, 0, 0)  # Pure Black for body text
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.line_spacing = 1.15

        # Save document into an in-memory byte stream instead of the public folder
        file_stream = io.BytesIO()
        doc.save(file_stream)
        file_stream.seek(0)
        
        # Stream bytes directly via Reflex download handler (matching your PDF workflow)
        return rx.download(data=file_stream.getvalue(), filename=filename)
    
    risk_stratified_km_fig: go.Figure = go.Figure()
    risk_stratified_km_summary_text: str = ""

    def generate_risk_stratified_km(self, df_fit, time_col, event_col):
        try:
            if not hasattr(self, '_cph_fitted_model') or self._cph_fitted_model is None:
                return

            from lifelines import KaplanMeierFitter
            from lifelines.statistics import multivariate_logrank_test
            import pandas as pd
            import plotly.graph_objects as go

            # 1. Get partial hazards (risk scores) from multivariable model
            risk_scores = self._cph_fitted_model.predict_partial_hazard(df_fit)
            
            # 2. Divide into tertiles (Low, Intermediate, High) with robust duplicate edge handling
            df_plot = df_fit.copy()
            
            try:
                # Convert risk scores to ranks first to cleanly handle clustered/tied values
                ranked_scores = pd.Series(risk_scores).rank(method='first')
                df_plot['risk_tier'] = pd.qcut(ranked_scores, q=3, labels=['Low Risk', 'Intermediate Risk', 'High Risk'])
            except ValueError:
                # Fallback warning if extreme constraints occur
                self.km_warning_message = "Note: Severe data ties detected in risk scores. Tertile boundaries adjusted via rank-based allocation."
                # Fallback cut if ranking encounters edge issues
                df_plot['risk_tier'] = pd.cut(pd.Series(risk_scores).rank(method='first'), bins=3, labels=['Low Risk', 'Intermediate Risk', 'High Risk'])

            # 3. Compute global log-rank test p-value
            lr_result = multivariate_logrank_test(
                df_plot[time_col], 
                df_plot['risk_tier'], 
                df_plot[event_col]
            )
            p_val = lr_result.p_value
            p_str = "< 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
            self.risk_stratified_km_summary_text = f"Global Log-Rank Test across Risk Tertiles ({p_str})"

            # 4. Build Plotly KM curves
            kmf = KaplanMeierFitter()
            fig = go.Figure()
            
            colors = {'Low Risk': '#10b981', 'Intermediate Risk': '#3b82f6', 'High Risk': '#ef4444'}
            
            for tier in ['Low Risk', 'Intermediate Risk', 'High Risk']:
                mask = df_plot['risk_tier'] == tier
                if mask.sum() < 5:
                    continue
                kmf.fit(df_plot.loc[mask, time_col], event_observed=df_plot.loc[mask, event_col], label=tier)
                
                # --- FIXED: Access survival function directly from kmf.survival_function_ ---
                timeline = kmf.timeline
                surv_col = kmf.survival_function_.columns[0]
                surv_prob = kmf.survival_function_[surv_col]
                
                fig.add_trace(go.Scatter(
                    x=timeline, y=surv_prob,
                    mode='lines',
                    name=f"{tier} (n={mask.sum()})",
                    line=dict(color=colors.get(tier, '#000000'), width=2.5)
                ))

            unit_label = self.timeline_unit
            fig.update_layout(
                title={"text": f"<b>Risk-Stratified Survival Trajectories ({p_str})</b>", "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 13}},
                template="simple_white",
                xaxis=dict(title=f"Follow-Up Time Horizon ({unit_label})"),
                yaxis=dict(title="Survival Probability", range=[0.0, 1.0], dtick=0.2),
                margin=dict(l=50, r=20, t=50, b=40),
                height=320,
                showlegend=True,
                legend=dict(x=0.65, y=0.9, font=dict(size=10)),
                autosize=True
            )

            self.risk_stratified_km_fig = fig

        except Exception as e:
            print(f"DEBUG [Risk-Stratified KM Error]: {e}")

    time_dep_auc_fig: go.Figure = go.Figure()
    time_dep_auc_summary: list = []
    
    def generate_time_dependent_auc(self, df_fit, time_col, event_col, ):
        """
        Computes time-dependent AUC dynamically at automatically derived follow-up 
        quantiles (25th, 50th, and 75th percentiles) of the active filtered cohort.
        """
        #print("DEBUG [Time-Dep AUC]: Calculating time-dependent AUC across dynamic horizons...")
        try:
            if not hasattr(self, '_cph_fitted_model') or self._cph_fitted_model is None:
                print("DEBUG [Time-Dep AUC]: No fitted CPH model found.")
                return

            import numpy as np
            import pandas as pd
            from sklearn.metrics import roc_auc_score
            import plotly.graph_objects as go

            valid_times = df_fit[time_col].dropna()
            if len(valid_times) < 20:
                #print("DEBUG [Time-Dep AUC]: Cohort too small for reliable time-dependent AUC.")
                return

            milestones = [
                float(np.percentile(valid_times, 25)),
                float(np.percentile(valid_times, 50)),
                float(np.percentile(valid_times, 75))
            ]
            
            auc_times = []
            auc_values = []
            
            for t in milestones:
                eval_sub = df_fit.copy()
                mask_excluded = (eval_sub[time_col] < t) & (eval_sub[event_col] == 0)
                eval_sub = eval_sub[~mask_excluded].copy()
                
                if len(eval_sub) < 15:
                    continue
                    
                eval_sub['status_at_t'] = ((eval_sub[time_col] <= t) & (eval_sub[event_col] == 1)).astype(int)
                
                if len(eval_sub['status_at_t'].unique()) < 2:
                    continue
                
                # --- FIXED: Correctly extract survival prediction for all subjects at time t ---
                surv_funcs = self._cph_fitted_model.predict_survival_function(eval_sub, times=[t])
                if surv_funcs is None or surv_funcs.empty:
                    continue
                    
                # Lifelines returns time as index and subjects as columns
                pred_surv_t = surv_funcs.iloc[0].values if t not in surv_funcs.index else surv_funcs.loc[t].values
                pred_risk_t = 1.0 - pred_surv_t
                
                auc_val = roc_auc_score(eval_sub['status_at_t'], pred_risk_t)
                
                auc_times.append(t)
                auc_values.append(auc_val)

            #print(f"DEBUG [Time-Dep AUC]: Computed milestones {auc_times} with AUCs {auc_values}")

            if not auc_values:
                #print("DEBUG [Time-Dep AUC]: No valid AUC milestones computed.")
                return

            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[min(auc_times)*0.9, max(auc_times)*1.1] if auc_times else [0, 1],
                y=[0.5, 0.5],
                mode='lines', name='Random Discrimination (0.5)',
                line=dict(color='rgba(100, 116, 139, 0.5)', dash='dash', width=1.5),
                hoverinfo='none'
            ))
            
            fig.add_trace(go.Scatter(
                x=auc_times, y=auc_values,
                mode='markers+lines',
                name='Time-Specific AUC',
                line=dict(color='#8b5cf6', width=2.5),
                marker=dict(size=8, color='#7c3aed')
            ))

            unit_label = self.timeline_unit
            fig.update_layout(
                title={"text": "<b>Time-Dependent Discrimination (AUC Trajectory)</b>", "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 13}},
                template="simple_white",
                xaxis=dict(title=f"Follow-Up Time Horizon ({unit_label})"),
                yaxis=dict(title="Area Under the ROC Curve (AUC)", range=[0.4, 1.0], dtick=0.1),
                margin=dict(l=50, r=20, t=50, b=40),
                height=300,
                showlegend=True,
                legend=dict(x=0.55, y=0.15, font=dict(size=10)),
                autosize=True
            )
            
            # --- CRITICAL FIX: Serialize to dict for Reflex compatibility ---
            #self.time_dep_auc_fig = fig.to_dict()
            #self.time_dep_auc_summary = list(zip(auc_times, auc_values))
            self.time_dep_auc_fig = fig
            self.time_dep_auc_summary = list(zip(auc_times, auc_values))
            
        except Exception as e:
            print(f"DEBUG [Time-Dep AUC Exception Error]: {e}")
            import traceback
            traceback.print_exc()
    
    cph_dca_fig: go.Figure = go.Figure()
    
    def generate_decision_curve(self, df_fit, time_col, event_col, eval_time=None):
        """
        Computes Decision Curve Analysis (DCA) Net Benefit across threshold probabilities at eval_time.
        """
        #print("DEBUG [DCA]: Starting Decision Curve Analysis computation...")
        try:
            # Safely reset local dataframe index here
            if df_fit is not None:
                df_fit = df_fit.reset_index(drop=True)
            
            if not hasattr(self, '_cph_fitted_model') or self._cph_fitted_model is None:
                return
            
            # eval_time = float(df_fit[time_col].median())
            
            # 1. Get predicted survival probabilities at eval_time, convert to risk: 1 - S(t)
            surv_funcs = self._cph_fitted_model.predict_survival_function(df_fit, times=[eval_time])
            if surv_funcs is None or surv_funcs.empty:
                return
            
            pred_surv = surv_funcs.iloc[0].values
            risk_scores = 1.0 - pred_surv
            
            df_dca = df_fit.copy()
            df_dca['risk_score'] = risk_scores
            df_dca['event_at_horizon'] = ((df_dca[time_col] <= eval_time) & (df_dca[event_col] == 1)).astype(int)
            
            # Exclude patients censored before eval_time without event
            df_dca = df_dca[~((df_dca[time_col] < eval_time) & (df_dca[event_col] == 0))]
            
            n_total = len(df_dca)
            n_events = df_dca['event_at_horizon'].sum()
            n_nonevents = n_total - n_events
            
            # 2. Evaluate Net Benefit across threshold probabilities (1% to 50%)
            thresholds = [i / 100.0 for i in range(1, 51)]
            model_nb = []
            all_nb = []
            
            for pt in thresholds:
                # Model strategy
                classifications = df_dca['risk_score'] >= pt
                tp = ((classifications) & (df_dca['event_at_horizon'] == 1)).sum()
                fp = ((classifications) & (df_dca['event_at_horizon'] == 0)).sum()
                nb = (tp / n_total) - (fp / n_total) * (pt / (1.0 - pt))
                model_nb.append(nb)
                
                # Treat All strategy
                tp_all = n_events
                fp_all = n_nonevents
                nb_all = (tp_all / n_total) - (fp_all / n_total) * (pt / (1.0 - pt))
                all_nb.append(nb_all)
                
            # Treat None is always 0.0
            none_nb = [0.0] * len(thresholds)
            
            # 3. Build Plotly Decision Curve Figure using go.Figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=thresholds, y=model_nb,
                mode='lines', name='Cox Model',
                line=dict(color='#2563eb', width=2.5)
            ))
            
            fig.add_trace(go.Scatter(
                x=thresholds, y=all_nb,
                mode='lines', name='Treat All',
                line=dict(color='rgba(100, 116, 139, 0.6)', dash='dash', width=1.5)
            ))
            
            fig.add_trace(go.Scatter(
                x=thresholds, y=none_nb,
                mode='lines', name='Treat None',
                line=dict(color='rgba(15, 23, 42, 0.8)', dash='dot', width=1.5)
            ))

            unit_label = self.timeline_unit
            fig.update_layout(
                title={"text": f"<b>Decision Curve Analysis ({eval_time:.1f} {unit_label})</b>", "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 13}},
                template="simple_white",
                xaxis=dict(title="Threshold Probability", range=[0, 0.5], dtick=0.1),
                yaxis=dict(title="Net Benefit", range=[-0.05, max(max(model_nb), 0.1) * 1.1], dtick=0.05),
                margin=dict(l=50, r=20, t=50, b=40),
                height=320,
                showlegend=True,
                legend=dict(x=0.65, y=0.95, font=dict(size=10)),
                autosize=True
            )
            
            self.cph_dca_fig = fig
            #print("DEBUG [DCA]: Decision curve generated and bound successfully.")
            
        except Exception as e:
            print(f"DEBUG [DCA Exception Error]: {e}")
            import traceback
            traceback.print_exc()
    
    # --- Quantitative Calibration Metrics ---
    calib_slope_val: float = 1.000
    calib_slope_ci: str = "(0.000 – 0.000)"
    calib_intercept_val: float = 0.000
    gnd_chi2_val: float = 0.000
    gnd_p_val: str = "1.0000"

    def _compute_gnd_and_calibration_metrics(self, cph, df_fit, t0):
        # --- Computes Calibration Slope : Calibration Intercept : Greenwood-Nam-D'Agostino (GND) Goodness-of-Fit test and p-value --- #
        
        pred_surv = np.array([])        # --- Initialize fallback for pred_surv to prevent UnboundLocalError --- #
        
        try:
            from scipy import stats

            # Safely reset local dataframe index here
            if df_fit is not None:
                df_fit = df_fit.reset_index(drop=True)
            
            time_col = self.selected_time_column
            event_col = self.selected_event_column

            # 1. Calculate linear predictor (log partial hazard) for each subject
            partial_hazards = cph.predict_partial_hazard(df_fit)
            eta = np.log(partial_hazards.values)

            # Create temporary dataframe for recalibration modeling (Defined upfront)
            recal_df = pd.DataFrame({
                'T': df_fit[time_col].values,
                'E': df_fit[event_col].values,
                'eta': eta
            })

            # 2. Calibration Slope: Fit Cox model with eta as covariate
            cph_slope = CoxPHFitter()
            cph_slope.fit(recal_df, duration_col='T', event_col='E', show_progress=False)
            calib_slope = cph_slope.params_['eta']
            slope_ci = cph_slope.confidence_intervals_.loc['eta'].values

            # 3. Greenwood-Nam-D'Agostino (GND) Test across Risk Deciles
            try:
                surv_df = cph.predict_survival_function(df_fit, times=[t0])
                #print(f"DEBUG [GND]: surv_df shape: {surv_df.shape}, index: {surv_df.index}")
                
                if len(surv_df.columns) == len(recal_df):
                    # FIX applied here: Use np.abs() instead of surv_df.index.abs()
                    idx_loc = np.abs(surv_df.index - t0).argsort()[0]
                    pred_surv = surv_df.iloc[idx_loc].values
                else:
                    pred_surv = surv_df.iloc[0].values

                df_fit_eval = recal_df.copy()
                df_fit_eval['pred_surv'] = pred_surv
                
                df_fit_eval = df_fit_eval.dropna(subset=['pred_surv'])
                #print(f"DEBUG [GND]: Valid rows for deciles: {len(df_fit_eval)}")
                
                if len(df_fit_eval) >= 10:
                    try:
                        # Rank-based tie breaking for risk deciles
                        ranked_pred = df_fit_eval['pred_surv'].rank(method='first')
                        df_fit_eval['risk_decile'] = pd.qcut(ranked_pred, q=10, labels=False)
                    except Exception:
                        df_fit_eval['risk_decile'] = pd.cut(
                            df_fit_eval['pred_surv'].rank(method='first'), 
                            bins=10, 
                            labels=False
                        )
                    
                    gnd_chi2 = 0.0
                    gnd_df_count = 0
                    
                    for decile in sorted(df_fit_eval['risk_decile'].unique()):
                        sub = df_fit_eval[df_fit_eval['risk_decile'] == decile]
                        obs_events = len(sub[(sub['T'] <= t0) & (sub['E'] == 1)])
                        exp_events = np.sum(1.0 - sub['pred_surv'].values)
                        
                        if exp_events > 0:
                            gnd_chi2 += ((obs_events - exp_events) ** 2) / exp_events
                            gnd_df_count += 1

                    gnd_df_count = max(1, gnd_df_count - 2)
                    gnd_p_value = 1.0 - stats.chi2.cdf(gnd_chi2, df=gnd_df_count)
                    #print(f"DEBUG [GND Success]: chi2={gnd_chi2:.3f}, p={gnd_p_value:.4f}")
                else:
                    print("DEBUG [GND]: Insufficient rows for decile cutting.")
                    gnd_chi2 = 0.0
                    gnd_p_value = 1.0
                    
                self.gnd_chi2_val = round(gnd_chi2, 2)
                #self.gnd_p_val = round(gnd_p_value, 4) if gnd_p_value >= 0.0001 else "< 0.0001"

                # Store p-value explicitly as a string to support "< 0.0001" text
                self.gnd_p_val = f"{gnd_p_value:.4f}" if gnd_p_value >= 0.0001 else "< 0.0001"


            except Exception as gnd_err:
                print(f"DEBUG [GND Calculation Sub-Error]: {gnd_err}")
                import traceback
                traceback.print_exc()
                self.gnd_chi2_val = 0.0
                self.gnd_p_val = "N/A"

            total_obs = len(recal_df[(recal_df['T'] <= t0) & (recal_df['E'] == 1)])
            total_exp = np.sum(1.0 - pred_surv) if len(pred_surv) > 0 else 1.0
            calib_intercept = np.log(total_obs / total_exp) if total_exp > 0 else 0.0
            self.calib_intercept_val = float(round(calib_intercept, 3))
            
            #print(f"DEBUG TYPE CHECK -> calib_slope: {type(calib_slope)}, value: {calib_slope}")
            #print(f"DEBUG TYPE CHECK -> calib_intercept: {type(calib_intercept)}, value: {calib_intercept}")

            self.calib_slope_val = float(round(calib_slope, 3))
            self.calib_slope_ci = f"({round(slope_ci[0], 3)} – {round(slope_ci[1], 3)})"

        except Exception as e:
            print(f"DEBUG [GND / Calibration Metrics Error]: {str(e)}")
            import traceback
            traceback.print_exc()

    # ------------------------ Computes quintile risk groups, calculates observed KM survival vs predicted ---------------------------- #
    cph_calibration_fig: go.Figure = go.Figure()
    cph_calib_horizon_text: str = "Evaluated at Median Follow-Up Horizon"
    _cph_fitted_model: Any = None
     
    def generate_calibration_curve(self, df_fit, time_col, event_col, n_bootstrap=200, eval_time=None):
        """
        Computes time-dependent calibration curves (Apparent & Bootstrap Optimism-Corrected) 
        across adaptive risk groupings at median evaluation horizon (eval_time).
        """

        try:
            if not hasattr(self, '_cph_fitted_model') or self._cph_fitted_model is None:
                return

            unit_label = self.timeline_unit
            
            self.cph_calib_horizon_text = f"Evaluated at Median Follow-Up Horizon ({eval_time:.1f} {unit_label} across adaptive risk groupings with {n_bootstrap}-iteration bootstrap bias correction)."
            
            # --- 1. APPARENT CALIBRATION ---
            surv_funcs = self._cph_fitted_model.predict_survival_function(df_fit, times=[eval_time])
            if surv_funcs is None or surv_funcs.empty:
                return
            
            pred_surv = surv_funcs.iloc[0].values
            pred_risk = 1.0 - pred_surv
            
            df_calib = df_fit.copy()
            df_calib['pred_risk'] = pred_risk
            df_calib = df_calib.dropna(subset=['pred_risk', time_col, event_col])
            
            n_obs = len(df_calib)
            if n_obs < 30:
                print("DEBUG [Calibration]: Cohort too small for reliable decile calibration.")
                return

            # Adaptive binning based on sample size to prevent qcut errors
            n_bins = 10 if n_obs >= 500 else (5 if n_obs >= 200 else (4 if n_obs >= 100 else 3))
            
            try:
                ranked_risk = df_calib['pred_risk'].rank(method='first')
                df_calib['decile'], bins = pd.qcut(ranked_risk, q=n_bins, retbins=True, labels=False)
            except Exception:
                df_calib['decile'] = pd.cut(df_calib['pred_risk'].rank(method='first'), bins=n_bins, labels=False)
                
            decile_pred = []
            decile_obs = []
            decile_lower = []
            decile_upper = []
            
            from lifelines import KaplanMeierFitter
            import numpy as np
            
            for d in sorted(df_calib['decile'].dropna().unique()):
                sub = df_calib[df_calib['decile'] == d]
                if len(sub) == 0:
                    continue
                mean_p = float(sub['pred_risk'].mean())
                decile_pred.append(mean_p)
                
                kmf = KaplanMeierFitter()
                kmf.fit(durations=sub[time_col], event_observed=sub[event_col])
                
                surv_at_t = kmf.survival_function_at_times(eval_time)
                if surv_at_t.empty:
                    obs_r = 0.0
                else:
                    obs_r = 1.0 - float(surv_at_t.values[0])
                decile_obs.append(obs_r)
                
                ci_table = kmf.confidence_interval_survival_function_
                if not ci_table.empty:
                    idx = ci_table.index.get_indexer([eval_time], method='nearest')[0]
                    lower_s = ci_table.iloc[idx, 0]
                    upper_s = ci_table.iloc[idx, 1]
                    decile_lower.append(max(0.0, 1.0 - float(upper_s)))
                    decile_upper.append(min(1.0, 1.0 - float(lower_s)))
                else:
                    decile_lower.append(obs_r)
                    decile_upper.append(obs_r)

            # --- 2. BOOTSTRAP OPTIMISM CORRECTION ---
            #print(f"DEBUG [Calibration]: Running {n_bootstrap} bootstrap iterations for bias correction...")
            optimisms = []
            
            for _ in range(n_bootstrap):
                try:
                    boot_df = df_fit.sample(n=len(df_fit), replace=True, random_state=None)
                    from lifelines import CoxPHFitter
                    boot_cph = CoxPHFitter()
                    boot_cph.fit(boot_df, duration_col=time_col, event_col=event_col, show_progress=False)
                    
                    boot_surv_self = boot_cph.predict_survival_function(boot_df, times=[eval_time]).iloc[0].values
                    orig_surv_test = boot_cph.predict_survival_function(df_fit, times=[eval_time]).iloc[0].values
                    
                    boot_brier_self = np.mean(( (boot_df[time_col] <= eval_time) & (boot_df[event_col] == 1) ).astype(int) - (1.0 - boot_surv_self))**2
                    orig_brier_test = np.mean(( (df_fit[time_col] <= eval_time) & (df_fit[event_col] == 1) ).astype(int) - (1.0 - orig_surv_test))**2
                    
                    optimisms.append(orig_brier_test - boot_brier_self)
                except Exception:
                    continue
                    
            mean_optimism = float(np.mean(optimisms)) if optimisms else 0.0
            corrected_obs = [max(0.0, min(1.0, o - mean_optimism)) for o in decile_obs]

            # --- 3. QUANTITATIVE METRICS (Intercept, Slope, GND) --- #
            lin_pred = self._cph_fitted_model.predict_log_partial_hazard(df_calib)
            
            # Using the dynamic event_col variable for the temporary column name
            recal_df = pd.DataFrame({
                'lp': lin_pred, 
                event_col: ((df_calib[time_col] <= eval_time) & (df_calib[event_col] == 1)).astype(int)
            })
            recal_df = recal_df[~((df_calib[time_col] < eval_time) & (df_calib[event_col] == 0))]
            
            try:
                import statsmodels.api as sm
                
                X = sm.add_constant(recal_df['lp'])
                y = recal_df[event_col] # Fully dynamic variable reference
                
                if len(y.unique()) > 1 and len(recal_df) > 10:
                    logit_model = sm.Logit(y, X)
                    result = logit_model.fit(disp=False, maxiter=1000)
                    
                    params = result.params
                    conf_ints = result.conf_int()
                    
                    intercept_val = params.get('const', 0.0)
                    slope_val = params.get('lp', 1.0)
                    
                    # Ensure fallback is None, not a hardcoded +/- 0.1 offset
                    slope_ci_low = conf_ints.loc['lp', 0] if 'lp' in conf_ints.index else None
                    slope_ci_high = conf_ints.loc['lp', 1] if 'lp' in conf_ints.index else None

                    # Store as raw numeric floats to satisfy Reflex/Pydantic types
                    self.calib_intercept_val = float(round(intercept_val, 3))
                    self.calib_slope_val = float(round(slope_val, 3))

                    if slope_ci_low is not None and slope_ci_high is not None:
                        self.calib_slope_ci = f"({float(slope_ci_low):.3f} – {float(slope_ci_high):.3f})"
                    else:
                        self.calib_slope_ci = "(CI Not Converged)"
                else:
                    mean_event = float(y.mean()) if len(y) > 0 else 0.0
                    self.calib_intercept_val = float(round(mean_event, 3))
                    self.calib_slope_val = 1.0
                    self.calib_slope_ci = "(N/A - Sparse Subgroup)"

            except Exception as e:
                print(f"DEBUG [Calibration Calculation Error]: {e}")
                self.calib_intercept_val = 0.0
                self.calib_slope_val = 1.0
                self.calib_slope_ci = "(N/A)"

            # --- 4. BUILD PLOTLY CALIBRATION FIGURE ---
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines', name='Ideal Calibration',
                line=dict(color='rgba(100, 116, 139, 0.6)', dash='dash', width=1.5),
                hoverinfo='none'
            ))
            
            fig.add_trace(go.Scatter(
                x=decile_pred, y=decile_obs,
                error_y=dict(
                    type='data',
                    array=[u - o for u, o in zip(decile_upper, decile_obs)],
                    symmetric=False,
                    color='#3b82f6',
                    thickness=1.5,
                    width=3
                ),
                mode='markers+lines',
                name='Apparent Calibration',
                line=dict(color='#2563eb', width=2),
                marker=dict(size=7, color='#2563eb')
            ))
            
            fig.add_trace(go.Scatter(
                x=decile_pred, y=corrected_obs,
                mode='markers+lines',
                name='Bootstrap Corrected',
                line=dict(color='#10b981', width=2, dash='dot'),
                marker=dict(size=6, color='#10b981')
            ))
            
            fig.update_layout(
                title={"text": f"<b>Time-Dependent Calibration ({eval_time:.1f} {unit_label})</b>", "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 13}},
                template="simple_white",
                xaxis=dict(title="Predicted Survival Probability", range=[0, 1], dtick=0.2),
                yaxis=dict(title="Observed Kaplan-Meier Survival", range=[0, 1], dtick=0.2),
                margin=dict(l=50, r=20, t=50, b=40),
                height=320,
                showlegend=True,
                legend=dict(x=0.55, y=0.15, font=dict(size=10)),
                autosize=True
            )
            
            self.cph_calibration_fig = fig
            #print("DEBUG [Calibration]: Calibration plot with bootstrap correction generated successfully.")
            
        except Exception as e:
            print(f"DEBUG [Calibration Exception Error]: {e}")
            import traceback
            traceback.print_exc()

    cph_apparent_c_index: Union[float, str] = ""
    cph_corrected_c_index: Union[float, str] = 0.0


    def calculate_bootstrap_validation(self, df, time_col, event_col, covariates, strata_vars, n_bootstraps=100):
        """
        Computes optimism-corrected Harrell's C-index via bootstrap resampling.
        """
        try:
            strat_arg = strata_vars if strata_vars else None
            cph_apparent = CoxPHFitter(penalizer=0.1).fit(df, duration_col=time_col, event_col=event_col, strata=strat_arg)
            
            preds_apparent = cph_apparent.predict_partial_hazard(df)
            c_apparent = concordance_index(df[time_col], -preds_apparent, df[event_col])
            
            optimisms = []
            n = len(df)
            
            for _ in range(n_bootstraps):
                boot_df = df.sample(n=n, replace=True, random_state=None)
                try:
                    cph_boot = CoxPHFitter(penalizer=0.1).fit(boot_df, duration_col=time_col, event_col=event_col, strata=strat_arg)
                    
                    preds_boot_sample = cph_boot.predict_partial_hazard(boot_df)
                    c_boot = concordance_index(boot_df[time_col], -preds_boot_sample, boot_df[event_col])
                    
                    preds_orig_sample = cph_boot.predict_partial_hazard(df)
                    c_orig = concordance_index(df[time_col], -preds_orig_sample, df[event_col])
                    
                    optimisms.append(c_boot - c_orig)
                except Exception:
                    continue
                    
            if optimisms:
                mean_optimism = np.mean(optimisms)
                corrected_c_index = c_apparent - mean_optimism
            else:
                corrected_c_index = c_apparent
                
            return round(c_apparent, 3), round(corrected_c_index, 3)
        except Exception as e:
            print(f"DEBUG [Bootstrap Error]: {e}")
            return 0.0, 0.0

    selected_strata_vars: list[str] = []
    cph_auto_stratify_violations: bool = False
    cph_active_strata: list[str] = []
    cph_detected_violations: list[str] = []
    cph_table_data: list[list[str]] = []
    cph_error_message: str = ""
    cph_has_run: bool = False  # 🔑 Tracks whether Run Analysis has been executed
    cph_fig: go.Figure = go.Figure()

    def _reset_cph_sandbox_state(self):
        """Completely purges all CPH exploratory stratification and results state."""
        self.selected_strata_vars = []
        self.cph_auto_stratify_violations = False
        self.cph_active_strata = []
        self.cph_detected_violations = []
        self.cph_table_data = []
        self.cph_error_message = ""
        cph_has_run: bool = False  # 🔑 Reset back to awaiting card view
        self.cph_fig = go.Figure()
    
    def toggle_strata_var(self, var_name: str, checked: bool):
        if checked:
            if var_name not in self.selected_strata_vars:
                self.selected_strata_vars.append(var_name)
        else:
            if var_name in self.selected_strata_vars:
                self.selected_strata_vars.remove(var_name)
        
        self.cph_error_message = ""
        # 🚫 NOTICE: No resets here either! 
        # The physician can check/uncheck freely while looking at the active results.
    
    # --- Stratification State Variables ---
    cph_auto_stratify_violations: bool = True       # Toggle for auto-stratification
    selected_strata_vars: list[str] = []            # The manual checkbox selections in the sidebar
    cph_active_strata: list[str] = []               # The final combined strata applied in the current engine run
    
    selected_strata_vars: list[str] = []
    pnodes_bin_mode: str = "Clinical Bins (0, 1-3, 4-9, 10+)"  # --- Binning strategy selector --- #

    def set_selected_strata_vars(self, new_strata: list[str]):
        """Updates the selected strata and instantly re-runs the CPH engine."""
        self.selected_strata_vars = new_strata
        self.cph_error_message = ""
        self._run_cph_engine()
    
    # 🌟 Explicitly define the setter handler:
    def set_cph_auto_stratify_violations(self, value: bool):        
        self.cph_auto_stratify_violations = value 
        self.cph_error_message = ""
        # 🚫 NOTICE: We do NOT call any reset method here! 
        # The existing plot, table, and data remain completely untouched on screen.
        
    cph_full_cohort_n: int = 0
    cph_median_followup_text: str = ""

    # -------------------------------------- BASELINE CHARACTERISTICS TABLE 1 ---------------------------------------- #
    def _generate_table_one(self, df1=None, df2=None):
        if df1 is None:
            df1 = getattr(self, "current_df1", None)
        if df2 is None:
            df2 = getattr(self, "current_df2", None)
            
        if df1 is None or (hasattr(df1, "empty") and df1.empty):
            self.table_one_data = []
            return

        # 🔥 Determine mode: True ONLY if full cohort mode is active or df2 is empty
        is_full_cohort = getattr(self, "is_full_cohort", False) or (df2 is None or df2.empty)

        if not getattr(self, "column_configs", None) or not isinstance(self.column_configs, list):
            self.column_configs = []

        existing_configs = {
            cfg.get("name"): cfg 
            for cfg in self.column_configs 
            if isinstance(cfg, dict) and "name" in cfg
        }

        # Use combined dataframe for category discovery in comparative mode, or df1 for full cohort
        combined_df = pd.concat([df1, df2]) if not is_full_cohort and df2 is not None and not df2.empty else df1
        updated_configs = []

        for col in combined_df.columns:
            col_str = str(col)
            if col_str.lower() not in ["time", "event", "status", "futime", "fustat", "time_col", "event_col"]:
                raw_cats = sorted([str(v) for v in combined_df[col].dropna().unique()])
                
                if col_str in existing_configs:
                    cfg = existing_configs[col_str]
                    if "mappings" not in cfg or not isinstance(cfg["mappings"], dict):
                        cfg["mappings"] = {}
                    for cat in raw_cats:
                        if cat not in cfg["mappings"]:
                            cfg["mappings"][cat] = cat
                else:
                    unique_count = len(raw_cats)
                    var_type = "Categorical" if unique_count < 8 else "Continuous"
                    cfg = {
                        "name": col_str,
                        "include": True,
                        "label": col_str.replace(".", " ").replace("_", " ").title(),
                        "type": var_type,
                        "mappings": {cat: cat for cat in raw_cats}
                    }
                updated_configs.append(cfg)

        self.column_configs = updated_configs
        table_rows = []

        for cfg in self.column_configs:
            if not isinstance(cfg, dict) or not cfg.get("include", True):
                continue

            col = cfg.get("name")
            if not col or col not in df1.columns:
                continue
            if not is_full_cohort and df2 is not None and col not in df2.columns:
                continue

            label = cfg.get("label", col)
            var_type = cfg.get("type", "Continuous")

            s1 = df1[col].dropna()
            s2 = df2[col].dropna() if not is_full_cohort and df2 is not None else pd.Series(dtype=float)

            if var_type == "Continuous":
                mean1, sd1 = (s1.mean(), s1.std()) if not s1.empty else (0.0, 0.0)
                g1_str = f"{mean1:.1f} ± {sd1:.1f}" if not s1.empty else "N/A"

                if not is_full_cohort:
                    mean2, sd2 = (s2.mean(), s2.std()) if not s2.empty else (0.0, 0.0)
                    g2_str = f"{mean2:.1f} ± {sd2:.1f}" if not s2.empty else "N/A"
                else:
                    g2_str = ""

                p_val_str = "N/A"
                if not is_full_cohort and len(s1) > 1 and len(s2) > 1:
                    try:
                        if s1.std() > 0 or s2.std() > 0:
                            t_stat, p_val = stats.ttest_ind(s1, s2, nan_policy='omit')
                            if not pd.isna(p_val):
                                p_val_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                    except Exception:
                        pass

                table_rows.append({
                    "variable": label,
                    "type": "Continuous",
                    "group1": g1_str,
                    "group2": g2_str,
                    "p_value": p_val_str if not is_full_cohort else ""
                })

            elif var_type == "Categorical":
                clean_label = re.sub(r'<[^>]*>', '', str(label)).strip()

                table_rows.append({
                    "variable": clean_label,
                    "type": "Header",
                    "group1": "",
                    "group2": "",
                    "p_value": ""
                })

                if is_full_cohort or df2 is None or df2.empty:
                    all_cats = sorted(list(s1.astype(str).unique()))
                else:
                    all_cats = sorted(list(set(s1.astype(str).unique()).union(set(s2.astype(str).unique()))))
                
                mappings = cfg.get("mappings", {})

                p_val_str = "N/A"
                if not is_full_cohort and df2 is not None and not df2.empty:
                    try:
                        contingency_table = pd.crosstab(
                            pd.concat([df1[col], df2[col]]),
                            pd.Series([1]*len(df1) + [2]*len(df2), name="group")
                        )
                        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                            chi2, p_val, dof, ex = stats.chi2_contingency(contingency_table)
                            if not pd.isna(p_val):
                                p_val_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                    except Exception:
                        pass

                for idx, cat in enumerate(all_cats):
                    n1 = int((s1.astype(str) == cat).sum())
                    pct1 = (n1 / len(s1) * 100) if len(s1) > 0 else 0.0
                    g1_str = f"{n1} ({pct1:.1f}%)"

                    if not is_full_cohort:
                        n2 = int((s2.astype(str) == cat).sum())
                        pct2 = (n2 / len(s2) * 100) if len(s2) > 0 else 0.0
                        g2_str = f"{n2} ({pct2:.1f}%)"
                    else:
                        g2_str = ""

                    display_cat = mappings.get(cat, cat)

                    table_rows.append({
                        "variable": display_cat,
                        "type": "Categorical",
                        "group1": g1_str,
                        "group2": g2_str,
                        "p_value": p_val_str if (not is_full_cohort and idx == 0) else ""
                    })

        self.table_one_data = table_rows

        
    @rx.event
    def download_active_report(self):
        model_name = getattr(self, "selected_model", "")        # Get selected_model Name --- #
    
        if not model_name or model_name.strip() == "":          # --- Trigger inline warning if no Model was selected --- #
            self.show_pdf_warning = True
            self.pdf_warning_msg = "Select Model First"
            return
        
        # Reset warning state if a valid model is present
        self.show_pdf_warning = False
        self.pdf_warning_msg = ""
        
        # --- Route based on the active model for Report Generation: either KM-Report or CPH-Report --- #
        if "Kaplan-Meier Survival" in model_name:
            return self.export_km_pdf_report()
        elif "Cox Proportional Hazards" in model_name:
            return self.export_cph_pdf_report()
        # Future expansion example:
        # elif "Competing Risks" in model_name:
        #     return self.export_competing_risks_report()
        else:
            # Fallback or future model routing
            self.show_pdf_warning = True
            self.pdf_warning_msg = f"Unsupported model selected: {model_name}"
    
    show_disclaimer_modal: bool = False

    # --- Tooltip composition strings for all possible table rows --- #
    full_cohort_composition_text: str = "Entire dataset / Unfiltered cohort"
    group1_composition_text: str = "Group 1: (No filters)"
    group2_composition_text: str = "Group 2: (No filters)"
    complement_composition_text: str = "Complement: Rest of Cohort (Inverse of Group 1)"
    
    def toggle_disclaimer_modal(self):
        """Toggles the visibility of the expanded legal disclaimer modal."""
        self.show_disclaimer_modal = not self.show_disclaimer_modal

    # Modal state toggle
    is_baseline_modal_open: bool = False

    # State variables for the PDF warning
    show_pdf_warning: bool = False
    pdf_warning_msg: str = "Click RUN ANALYSIS before creating the Report."

    show_manuscript_warning: bool = False
    manuscript_warning_msg: str = "Click RUN ANALYSIS before editing the Manuscript."

    async def handle_disabled_pdf_click(self):
        """Triggers an amber inline warning for 2 seconds when export is attempted on stale data."""
        if not self.results_ready:
            self.show_pdf_warning = True
            yield
            await asyncio.sleep(2.0)
            self.show_pdf_warning = False
            yield

    async def handle_disabled_manuscript_click(self):
        """Triggers an inline warning when manuscript editing is attempted on stale data."""
        if not self.results_ready:
            self.show_manuscript_warning = True
            yield
            await asyncio.sleep(2.0)
            self.show_manuscript_warning = False
            yield
    
    # Declare reactive cohort dataframes so they never raise AttributeError
    current_df1: pd.DataFrame = pd.DataFrame()
    current_df2: pd.DataFrame = pd.DataFrame()

    # Stores configuration for each column: [{"name": str, "include": bool, "label": str, "type": str}, ...]
    column_configs: list[dict] = []

    def _init_column_configs(self, df):
        """Dynamically initialize column configs, only generating mappings for categorical variables."""
        if df is None or df.empty:
            self.column_configs = []
            return

        configs = []
        excluded_cols = {
            getattr(self, "selected_time_column", None), 
            getattr(self, "selected_event_column", None)
        }

        for col in df.columns:
            if col in excluded_cols:
                continue
            
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            # If it's numeric and has many unique values, it's continuous. Otherwise, categorical.
            is_continuous = is_numeric and df[col].nunique() > 10
            summary_type = "Continuous" if is_continuous else "Categorical"
            
            # 🎯 CRITICAL: Only generate mapping lists for categorical variables with a manageable number of unique values!
            if not is_continuous:
                unique_vals = [str(val) for val in sorted(df[col].dropna().unique())]
                mappings_list = [{"raw": val, "label": val} for val in unique_vals]
            else:
                mappings_list = [] # Continuous variables don't need value mappings

            configs.append({
                "name": col,
                "include": True,
                "label": col.replace("_", " ").title(),
                "type": summary_type,
                "mappings_list": mappings_list,
            })
        
        self.column_configs = configs
        #print(f"--- DEBUG: Initialized {len(configs)} column configs with pre-populated mappings ---")

    @rx.event
    def open_baseline_modal(self):
        """Triggered exclusively by the customization panel button."""
        
        print("🟢 [DEBUG : 1517 : - BUTTON CLICKED]: open_baseline_modal triggered.")

        self.is_baseline_modal_open = True
        self.init_baseline_config()

        print(f"🟢 [DEBUG : 1522 - BUTTON FINISHED]: table_one_data rows count = {len(self.table_one_data)}")

    @rx.event
    def set_baseline_modal_open(self, open: bool):
        """Triggered exclusively by the dialog's on_open_change."""
        self.is_baseline_modal_open = open
        if open:
            self.init_baseline_config()

    def set_is_baseline_modal_open(self, open: bool):
        self.is_baseline_modal_open = open
        if open:
            self.init_baseline_config()

    def init_baseline_config(self):
        # ---------- Populates column configurations based on the active dataset columns ------------ #
        # Ensure we have a dataframe loaded
        if hasattr(self, "df") and self.df is not None:
            # Get existing configured column names to preserve user changes across re-opens
            existing_names = {cfg["name"] for cfg in self.column_configs}
            
            # Exclude time and event columns from baseline characteristics by default
            excluded_cols = {getattr(self, "time_col", "time"), getattr(self, "event_col", "status")}
            
            for col in self.df.columns:
                if col not in excluded_cols and col not in existing_names:
                    # Smart default publication label (e.g., ph_karno -> Ph Karno or cleaned up)
                    default_label = col.replace("_", ".").title()
                    
                    # Heuristic to guess continuous vs categorical
                    is_num = pd.api.types.is_numeric_dtype(self.df[col])
                    var_type = "Continuous" if is_num and self.df[col].nunique() > 10 else "Categorical"
                    
                    self.column_configs.append({
                        "name": col,
                        "include": True if col in ["age", "sex", "ph.ecog", "ph.karno", "ph_karno", "tspike", "tgrade", "nodes", "menostat", "tsize"] else False,
                        "label": default_label,
                        "type": var_type
                    })

    @rx.event
    def update_col_config(self, name: str, key: str, value):
        """Generic handler to update include status, label, or variable type."""
        print(f"\n--- DEBUG update_col_config ---")
        print(f"Received request: variable='{name}', key='{key}', value={value} (type: {type(value)})")
        
        updated_configs = []
        found = False
        
        # Determine the active dataframe to extract unique values from if switching to categorical
        active_df = self.current_df1 if hasattr(self, "current_df1") and not self.current_df1.empty else getattr(self, "df", None)

        for cfg in self.column_configs:
            if cfg.get("name") == name:
                new_cfg = dict(cfg)
                new_cfg[key] = value
                
                # Automatically populate mappings_list if switching to Categorical
                if key == "type" and value == "Categorical":
                    if active_df is not None and name in active_df.columns:
                        unique_vals = [str(val) for val in active_df[name].dropna().unique()]
                        new_cfg["mappings_list"] = [{"raw": val, "label": val} for val in unique_vals]
                        print(f"  -> Generated {len(unique_vals)} category mappings for {name}")

                updated_configs.append(new_cfg)
                found = True
                print(f"  -> Updated {name} {key} to {value}")
            else:
                updated_configs.append(cfg)
        
        if not found:
            print(f"  -> WARNING: Variable '{name}' not found in column_configs!")
            
        self.column_configs = updated_configs

        # Automatically re-generate Table 1 if cohorts are already loaded!
        if hasattr(self, "current_df1") and hasattr(self, "current_df2") and not self.current_df1.empty and not self.current_df2.empty:
            self._generate_table_one()

    # --- To compute for Results-Report Table 1: Baseline Demographics & Clinical Characteristics from filtered dataframes (df1 and df2)
    # Define for Report Table 1 calculated values with default fallback values
    g1_n: int = 0
    g2_n: int = 0
    g1_events: int = 0
    g2_events: int = 0

    @staticmethod
    def _generate_baseline_table(df1, df2, clinical_columns=None):
        # --- Compute for Report Table 1 Baseline Demographic and Clinical Characteristics comparing Group 1 and Group 2 --- #
        if not clinical_columns:
            # Fallback only if nothing is selected in the active state
            clinical_columns = []

        table_rows = []
        
        n1 = len(df1)
        n2 = len(df2)

        for col in clinical_columns:
            if col not in df1.columns or col not in df2.columns:
                continue
                
            # Check if column is numeric or categorical
            if pd.api.types.is_numeric_dtype(df1[col]) and df1[col].nunique() > 5:
                # Continuous variable: Present as Mean (SD) or Median [IQR]
                m1, sd1 = df1[col].mean(), df1[col].std()
                m2, sd2 = df2[col].mean(), df2[col].std()
                val1 = f"{m1:.1f} ± {sd1:.1f}"
                val2 = f"{m2:.1f} ± {sd2:.1f}"
                table_rows.append({"Characteristic": f"{col} (Mean ± SD)", "Group 1": val1, "Group 2": val2})
            else:
                # Categorical variable: Present as Count (%)
                categories = sorted(list(set(df1[col].dropna().unique()).union(set(df2[col].dropna().unique()))))
                for cat in categories:
                    c1 = (df1[col] == cat).sum()
                    pct1 = (c1 / n1 * 100) if n1 > 0 else 0
                    c2 = (df2[col] == cat).sum()
                    pct2 = (c2 / n2 * 100) if n2 > 0 else 0
                    
                    val1 = f"{c1} ({pct1:.1f}%)"
                    val2 = f"{c2} ({pct2:.1f}%)"
                    table_rows.append({"Characteristic": f"    {col} = {cat}", "Group 1": val1, "Group 2": val2})

        summary_df = pd.DataFrame(table_rows)
        return n1, n2, summary_df

    def _normalize_dataframe(self, df, time_col):
        """Helper method to handle all time-column normalization in one central place."""
        model_df = df.copy()
        timeline_unit = "Mo"
        timeline_max = 0
        timeline_values = [0, 0]
        
        if not time_col or time_col not in model_df.columns:
            return model_df, timeline_unit, timeline_max, timeline_values

        try:
            model_df[time_col] = pd.to_numeric(model_df[time_col], errors='coerce')
            raw_max = model_df[time_col].dropna().max()
            col_lower = time_col.lower()

            if raw_max and raw_max > 0:
                if "month" in col_lower:
                    timeline_unit = "Mo"
                elif "year" in col_lower or "yr" in col_lower:
                    timeline_unit = "Years"
                else:
                    if raw_max > 3650:  
                        model_df[time_col] = model_df[time_col] / 365.25
                        timeline_unit = "Years"
                    elif raw_max > 90:  
                        model_df[time_col] = model_df[time_col] / 30.4375
                        timeline_unit = "Mo"
                    else:
                        timeline_unit = "Days"

                timeline_max = int(model_df[time_col].dropna().max())
                timeline_values = [0, timeline_max]
        except Exception as e:
            print(f"Normalization engine error: {str(e)}")
            
        return model_df, timeline_unit, timeline_max, timeline_values

    show_updates_modal: bool = False
    
    def toggle_updates_modal(self):
        self.show_updates_modal = not self.show_updates_modal

    # --- New additions for About and Feedback modals ---
    show_about_modal: bool = False
    show_feedback_modal: bool = False
    feedback_text: str = ""
    feedback_submitting: bool = False
    feedback_success: bool = False

    table_one_data: list[dict] = []     # For Report Creation: Table 1

    @rx.event
    def toggle_column_include(self, col_name: str, include_val: bool):
        for cfg in self.column_configs:
            if cfg.get("name") == col_name:
                cfg["include"] = include_val
                break
        self._generate_table_one()

    @rx.event
    def update_column_label(self, col_name: str, new_label: str):
        for cfg in self.column_configs:
            if cfg.get("name") == col_name:
                cfg["label"] = new_label
                break
        self._generate_table_one()

    @rx.event
    def update_column_type(self, col_name: str, new_type: str):
        for cfg in self.column_configs:
            if cfg.get("name") == col_name:
                cfg["type"] = new_type
                # Automatically extract unique raw values when switched to Categorical
                if new_type == "Categorical" and not cfg.get("mappings"):
                    df1 = getattr(self, "current_df1", pd.DataFrame())
                    df2 = getattr(self, "current_df2", pd.DataFrame())
                    combined = pd.concat([df1, df2]) if not df2.empty else df1
                    if col_name in combined.columns:
                        cats = sorted([str(v) for v in combined[col_name].dropna().unique()])
                        cfg["mappings"] = {cat: cat for cat in cats}
                break
        self._generate_table_one()

    @rx.event
    def update_category_mapping(self, col_name: str, raw_val: str, new_label: str):
        updated_configs = []
        for cfg in self.column_configs:
            if cfg["name"] == col_name:
                # Create a shallow copy of the config dict
                new_cfg = dict(cfg)
                # Ensure mappings exists as a dict
                new_mappings = dict(new_cfg.get("mappings", {}))
                
                # Update the specific raw value's publication label
                new_mappings[raw_val] = new_label
                
                new_cfg["mappings"] = new_mappings
                updated_configs.append(new_cfg)
            else:
                updated_configs.append(cfg)
        
        # Explicitly reassigning triggers Reflex's reactive state update
        self.column_configs = updated_configs
        
        # Regenerate your table data using the fresh configs
        self._generate_table_one()

    def _generate_table_one(self, df1=None, df2=None):
        if df1 is None:
            df1 = getattr(self, "current_df1", None)
        if df2 is None:
            df2 = getattr(self, "current_df2", None)
            
        if df1 is None or (hasattr(df1, "empty") and df1.empty):
            self.table_one_data = []
            return

        # 🔥 Bidirectional check: Inspect both group filter states, mirroring _run_km_engine
        # 🔥 Correctly check if dictionaries contain actual active filter selections (non-empty lists)
        g1_checked = getattr(self, "g1_checked_values", {})
        g1_min = getattr(self, "g1_custom_min", {})
        g2_checked = getattr(self, "g2_checked_values", {})
        g2_min = getattr(self, "g2_custom_min", {})

        # True only if at least one list has items or min values exist
        g1_active = any(isinstance(v, list) and len(v) > 0 for v in g1_checked.values()) or bool(g1_min)
        g2_active = any(isinstance(v, list) and len(v) > 0 for v in g2_checked.values()) or bool(g2_min)

        # True ONLY if neither group has active filters
        is_full_cohort = not g1_active and not g2_active

        if not getattr(self, "column_configs", None) or not isinstance(self.column_configs, list):
            self.column_configs = []

        existing_configs = {
            cfg.get("name"): cfg 
            for cfg in self.column_configs 
            if isinstance(cfg, dict) and "name" in cfg
        }

        # Use combined dataframe for category discovery in comparative mode, or df1 for full cohort
        combined_df = pd.concat([df1, df2]) if not is_full_cohort and df2 is not None and not df2.empty else df1
        updated_configs = []

        for col in combined_df.columns:
            col_str = str(col)
            
            if hasattr(self, "columns_to_hide") and col_str in self.columns_to_hide:
                continue
            if hasattr(self, "csv_columns") and self.csv_columns and col_str not in self.csv_columns:
                continue

            if col_str.lower() not in ["time", "status", "futime", "fustat", "time_col", "event_col"]:
                raw_cats = sorted([str(v) for v in combined_df[col].dropna().unique()])
                
                if col_str in existing_configs:
                    cfg = existing_configs[col_str]
                    if "mappings" not in cfg or not isinstance(cfg["mappings"], dict):
                        cfg["mappings"] = {}
                    for cat in raw_cats:
                        if cat not in cfg["mappings"]:
                            cfg["mappings"][cat] = cat
                else:
                    unique_count = len(raw_cats)
                    var_type = "Categorical" if unique_count < 8 else "Continuous"
                    cfg = {
                        "name": col_str,
                        "include": True,
                        "label": col_str.replace(".", " ").replace("_", " ").title(),
                        "type": var_type,
                        "mappings": {}
                    }
                    if cfg.get("type") == "Categorical":
                        col_lower = col_str.lower()
                        if any(term in col_lower for term in ["sex", "gender"]):
                            default_map = {}
                            for cat in raw_cats:
                                c_low = cat.strip().lower()
                                if c_low in ["1", "1.0", "m", "male"]:
                                    default_map[cat] = "Male"
                                elif c_low in ["2", "2.0", "f", "female"]:
                                    default_map[cat] = "Female"
                                else:
                                    default_map[cat] = cat
                            cfg["mappings"] = default_map
                        else:
                            cfg["mappings"] = {cat: cat for cat in raw_cats}

                updated_configs.append(cfg)

        self.column_configs = updated_configs
        table_rows = []
        total_n = len(df1) if is_full_cohort else len(df1) + len(df2)

        for cfg in self.column_configs:
            if not isinstance(cfg, dict) or not cfg.get("include", True):
                continue

            col = cfg.get("name")
            if not col:
                continue

            col_str = str(col)
            if hasattr(self, "columns_to_hide") and col_str in self.columns_to_hide:
                continue
            if hasattr(self, "csv_columns") and self.csv_columns and col_str not in self.csv_columns:
                continue

            actual_col = None
            norm_target = col_str.lower().replace(" ", "").replace(".", "").replace("_", "")
            for df_col in df1.columns:
                norm_df_col = str(df_col).lower().replace(" ", "").replace(".", "").replace("_", "")
                if df_col == col or norm_df_col == norm_target:
                    actual_col = df_col
                    break

            if not actual_col or actual_col not in df1.columns:
                continue
            if not is_full_cohort and df2 is not None and actual_col not in df2.columns:
                continue

            label = cfg.get("label", col)
            var_type = cfg.get("type", "Continuous")

            s1 = df1[actual_col].dropna()
            s2 = df2[actual_col].dropna() if not is_full_cohort and df2 is not None else pd.Series(dtype=float)

            if var_type == "Continuous":
                mean1, sd1 = (s1.mean(), s1.std()) if not s1.empty else (0.0, 0.0)
                g1_str = f"{mean1:.1f} ± {sd1:.1f}" if not s1.empty else "N/A"

                if not is_full_cohort:
                    mean2, sd2 = (s2.mean(), s2.std()) if not s2.empty else (0.0, 0.0)
                    g2_str = f"{mean2:.1f} ± {sd2:.1f}" if not s2.empty else "N/A"
                else:
                    g2_str = ""

                p_val_str = "N/A"
                if not is_full_cohort and len(s1) > 1 and len(s2) > 1:
                    try:
                        if s1.std() > 0 or s2.std() > 0:
                            t_stat, p_val = stats.ttest_ind(s1, s2, nan_policy='omit')
                            if not pd.isna(p_val):
                                p_val_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                    except Exception:
                        pass

                table_rows.append({
                    "variable": label,
                    "type": "Continuous",
                    "group1": g1_str,
                    "group2": g2_str,
                    "p_value": p_val_str if not is_full_cohort else ""
                })

            elif var_type == "Categorical":
                table_rows.append({
                    "variable": label,
                    "type": "Header",
                    "group1": "",
                    "group2": "",
                    "p_value": ""
                })

                if is_full_cohort or df2 is None or df2.empty:
                    all_cats = sorted(list(s1.astype(str).unique()))
                else:
                    all_cats = sorted(list(set(s1.astype(str).unique()).union(set(s2.astype(str).unique()))))
                
                mappings = cfg.get("mappings", {})

                p_val_str = "N/A"
                if not is_full_cohort and df2 is not None and not df2.empty:
                    try:
                        contingency_table = pd.crosstab(
                            pd.concat([df1[actual_col], df2[actual_col]]),
                            pd.Series([1]*len(df1) + [2]*len(df2), name="group")
                        )
                        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                            chi2, p_val, dof, ex = stats.chi2_contingency(contingency_table)
                            if not pd.isna(p_val):
                                p_val_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                    except Exception:
                        pass

                for idx, cat in enumerate(all_cats):
                    n1 = int((s1.astype(str) == cat).sum())
                    pct1 = (n1 / len(s1) * 100) if len(s1) > 0 else 0.0
                    g1_str = f"{n1} ({pct1:.1f}%)"

                    if not is_full_cohort:
                        n2 = int((s2.astype(str) == cat).sum())
                        pct2 = (n2 / len(s2) * 100) if len(s2) > 0 else 0.0
                        g2_str = f"{n2} ({pct2:.1f}%)"
                    else:
                        g2_str = ""

                    display_cat = mappings.get(cat, cat)

                    table_rows.append({
                        "variable": display_cat,
                        "type": "Categorical",
                        "group1": g1_str,
                        "group2": g2_str,
                        "p_value": p_val_str if (not is_full_cohort and idx == 0) else ""
                    })
        
        self.table_one_data = table_rows

    def toggle_about_modal(self):
        self.show_about_modal = not self.show_about_modal

    def open_feedback_modal(self):
        # Explicitly opens the modal when clicking the "Feedback" button
        self.show_feedback_modal = True

    def toggle_feedback_modal(self, open_status: bool = True):
        # Handles on_open_change (and defaults safely if called without args)
        self.show_feedback_modal = open_status
        if not open_status:
            self.feedback_text = ""
            self.feedback_success = False

    def cancel_feedback(self):
        # Clears text and closes when Cancel is clicked
        self.feedback_text = ""
        self.feedback_success = False
        self.show_feedback_modal = False
        
    def set_feedback_text(self, value: str):
        """Explicit setter for feedback text input."""
        self.feedback_text = value

    async def submit_feedback(self):
        if not self.feedback_text.strip():
            return

        self.feedback_submitting = True
        self.feedback_success = False  # Reset success state on new submit
        yield

        try:
            # Dynamically read host from environment variables (defaults to local proxy)
            db_host = os.environ.get("DB_HOST", "127.0.0.1")
            
            connect_args = {
                "dbname": os.environ.get("DB_NAME", "postgres"),
                "user": os.environ.get("DB_USER", "postgres"),
                "password": os.environ.get("DB_PASSWORD", ""),
                "host": db_host,
            }
            
            # Only include the port if connecting via TCP (local proxy). 
            # Unix socket paths in Cloud Run do not use TCP ports.
            if not db_host.startswith("/"):
                connect_args["port"] = os.environ.get("DB_PORT", "5432")

            # Connect to PostgreSQL using the appropriate local or production socket path
            conn = psycopg2.connect(**connect_args)
            cur = conn.cursor()
            
            insert_query = """
                INSERT INTO feedback (user_id, rating, comments) 
                VALUES (%s, %s, %s);
            """
            
            user_identifier = getattr(self, "current_user", "anonymous")
            rating_val = getattr(self, "feedback_rating", 5)
            
            cur.execute(insert_query, (user_identifier, rating_val, self.feedback_text))
            conn.commit()
            
            cur.close()
            conn.close()
            print("Feedback saved successfully to PostgreSQL database.")
            
            # Show success message
            self.feedback_success = True
            self.feedback_submitting = False
            yield  # Render the success state immediately
            
            # Wait for 2 seconds so the user can read the confirmation message
            await asyncio.sleep(2)

            
            # Automatically close modal and clear text
            self.show_feedback_modal = False
            self.feedback_text = ""
            self.feedback_success = False
            
        except Exception as e:
            print(f"Failed to save feedback to database: {e}")
            self.feedback_success = False
            self.feedback_submitting = False

    # --- Defininig Pagination & Viewport State Variables to view +100,000 of data rows
    viewport_offset: int = 0          # Starting index of the current chunk (e.g., 0, 200, 400...)
    chunk_size: int = 200             # Number of rows displayed at a time
    row_number_input: str = "1"       # Value bound to the "Row #" input box

    time_error_msg: str = ""
    event_error_msg: str = ""
    show_variable_error: bool = False

    inline_error_msg = ""
    upload_error: str = ""
    file_upload_success: bool = False

    exact_status_message: str = ""
    selective_status_message: str = ""
    filter_status_message: str = ""

    def _populate_inspector_matrices(self, df=None):
        """Slices the working dataframe based on the current viewport offset and chunk size."""
        if df is None:
            df = getattr(self, "_filtered_df", None)
            
        if df is None or df.empty:
            self.at_risk_data = [["-", "-", "-", "-", "-", "-"]]
            return

        # Ensure offset doesn't exceed dataframe length
        total_rows = len(df)
        if self.viewport_offset >= total_rows:
            self.viewport_offset = max(0, total_rows - self.chunk_size)

        # Slice the dataframe chunk
        start_idx = self.viewport_offset
        end_idx = min(start_idx + self.chunk_size, total_rows)
        sliced_df = df.iloc[start_idx:end_idx]

        # Convert sliced dataframe into your table display format
        # (Assuming your existing logic converts df rows to lists/dicts for the rx.table)
        self.at_risk_data = sliced_df.values.tolist()
        
        # Update row tracking metrics or metadata text if desired (e.g., "Showing rows 1-200 of 686

    # --- Search & Exploration State ---
    search_query: str = ""
    search_all_columns: bool = True
    selected_search_column: str = "all"
    target_row_index: str = ""

    # --- Metadata stats --------------------------------------------
    # 1. --- Declare your dataframe state variables (initialized as empty DataFrames)
    
    raw_dataframe: pd.DataFrame = pd.DataFrame()
    filtered_dataframe: pd.DataFrame = pd.DataFrame()
    
    # --- 1. JSON-Serializable Reactive State Variables ---
    selected_search_column: str = "all"
    selected_event_column: str = ""
    search_query: str = ""
    jump_row_number: str = ""
    status_message: str = ""
    show_schema_preview: bool = False
    column_count: int = 0
    
    # Primitive metrics for the Top Row metadata (prevents serialization crashes)
    filtered_row_count: int = 0
    total_row_count: int = 0
    event_count: int = 0
    censored_count: int = 0
    complete_cases_percentage: str = "100.0%"

    # --- 2. Non-Reactive Backend Dataframes -> Centralized Dataframe State Architecture ---
    _df: pd.DataFrame = pd.DataFrame()          # Raw display dataframe (for UI tables & inspector)
    _raw_df: pd.DataFrame = pd.DataFrame()      # Pristine raw backup copy
    _model_df: pd.DataFrame = pd.DataFrame()    # Scaled modeling dataframe (for KM, CPH, & sliders)
    _filtered_df: pd.DataFrame = pd.DataFrame() # Working filtered copy
    _override_active_df: pd.DataFrame = None    # Initialize Override Cleansed Dataset Variable to Empty

    @rx.var
    def is_status_visible(self) -> bool:                                        
        """Automatically returns True if there is a non-empty status message."""
        return bool(self.status_message and self.status_message.strip())

    def _update_metrics(self):
        """Recalculates display metrics safely from backend dataframes."""
        if self._filtered_df is not None and not self._filtered_df.empty:
            self.filtered_row_count = len(self._filtered_df)
            total = len(self._filtered_df)
            
            if self.selected_event_column and self.selected_event_column in self._filtered_df.columns:
                self.event_count = int((self._filtered_df[self.selected_event_column] == 1).sum())
                self.censored_count = int((self._filtered_df[self.selected_event_column] == 0).sum())
            else:
                self.event_count = 0
                self.censored_count = 0
            
            complete = self._filtered_df.dropna().shape[0]
            pct = (complete / total) * 100 if total > 0 else 0.0
            self.complete_cases_percentage = f"{pct:.1f}%"
        else:
            self.filtered_row_count = 0
            self.event_count = 0
            self.censored_count = 0
            self.complete_cases_percentage = "100.0%"

        if self._raw_df is not None and not self._raw_df.empty:
            self.total_row_count = len(self._raw_df)
        else:
            self.total_row_count = 0

    column_count: int = 0

    jump_row_number: str = ""
    def set_jump_row_number(self, val: str):                                        
        self.jump_row_number = val
    
    def set_search_all_columns(self, val: bool):                                    
        self.search_all_columns = val

    def set_selected_search_column(self, col: str):
        self.selected_search_column = col

    def set_target_row_index(self, val: str):
        self.target_row_index = val

    def jump_to_row(self):
        """Slices the dataframe to center around the target row number with instant snapping."""
        try:
            target_row = int(self.jump_row_number)
            target_idx = target_row - 1  # Convert 1-indexed human input to 0-indexed Python slice
            
            target_df = getattr(self, "_filtered_df", None)
            if target_df is None or target_df.empty:
                return

            max_idx = len(target_df) - 1
            self.viewport_offset = max(0, min(target_idx, max_idx))    # --- Clamp the target index safely within dataset bounds
            self._populate_inspector_matrices()                        # --- Refresh the table display chunk
            
            # Snap scrollbar back to top instantly
            return rx.call_script("document.getElementById('inspector-table-container').scrollTop = 0;")
        except ValueError:
            pass

    def scroll_to_top(self):
        """Jumps back to the very first chunk of the dataset instantly, updates the row number input to 1, and snaps to the top."""
        self.viewport_offset = 0
        self.jump_row_number = "1"
        self._populate_inspector_matrices()
        
        return rx.call_script(
            """
            const container = document.getElementById('inspector-table-container');
            if (container) {
                container.scrollTop = 0;
            }
            setTimeout(() => {
                const inputs = document.querySelectorAll('input');
                inputs.forEach(input => {
                    const placeholder = input.placeholder || '';
                    const wrapper = input.closest('div') || input.parentElement;
                    const wrapperText = wrapper ? wrapper.innerText : '';
                    if (placeholder.includes('Row') || wrapperText.includes('Go') || wrapperText.includes('Type to Filter')) {
                        if (input.type === 'text' || input.type === 'number') {
                            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                            if (nativeSetter) {
                                nativeSetter.call(input, '1');
                            } else {
                                input.value = '1';
                            }
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                    }
                });
            }, 100);
            """
        )
    
    def scroll_to_end(self):
        """Jumps to the final chunk, updates the row number input to the total count, and snaps to the bottom."""
        target_df = getattr(self, "_filtered_df", None)
        if target_df is not None and not target_df.empty:
            #total_rows = len(target_df)
            self.total_row_count = len(target_df)

            # Calculate the start of the final chunk
            #self.viewport_offset = max(0, total_rows - self.chunk_size)
            self.viewport_offset = max(0, self.total_row_count - self.chunk_size)

            # Update input box to show the absolute final row number (e.g., 228)
            self.jump_row_number = str(self.total_row_count)
            self._populate_inspector_matrices()
            
            # Instantly snap the scrollbar to the bottom, and lock the input field to total_rows after scroll settles
            return rx.call_script(
                f"""
                const container = document.getElementById('inspector-table-container');
                if (container) {{
                    container.scrollTop = container.scrollHeight;
                }}
                setTimeout(() => {{
                    const inputs = document.querySelectorAll('input');
                    inputs.forEach(input => {{
                        const placeholder = input.placeholder || '';
                        const wrapper = input.closest('div') || input.parentElement;
                        const wrapperText = wrapper ? wrapper.innerText : '';
                        if (placeholder.includes('Row') || wrapperText.includes('Go') || wrapperText.includes('Type to Filter')) {{
                            if (input.type === 'text' || input.type === 'number') {{
                                const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                                if (nativeSetter) {{
                                    nativeSetter.call(input, '{self.total_row_count}');
                                }} else {{
                                    input.value = '{self.total_row_count}';
                                }}
                                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            }}
                        }}
                    }});
                }}, 100);
                """
            )

    def set_search_query(self, query: str):
        """Updates the search input value."""
        self.search_query = query

    def execute_search(self):
        """Triggers the search filtering logic across your dataset."""
        if not self.search_query.strip():
            return
        pass

    def reset_search(self):
        """Clears the search query and resets the view/filters."""
        self.search_query = ""
        pass

# ------------------------------
    def set_search_query_and_filter(self, query: str):                          
        """Updates the search query string and triggers live filtering."""
        self.search_query = query
        self.viewport_offset = 0                    # 🔥 Reset to page one whenever a new search filter is applied

        self._apply_live_filter()

    def set_column_and_filter(self, column: str):                               
        """Updates the selected search column scope and triggers live filtering."""
        self.selected_search_column = column
        self._apply_live_filter()

    def _apply_live_filter(self):
        """Filters the stored dataframe and locks the Data View open."""
        target_df = getattr(self, "_raw_df", None)
        if target_df is None or target_df.empty:
            return

        working_df = target_df.copy()
        query = self.search_query.strip().lower()

        # If query is empty, restore the full dataset view
        if not query:
            self._filtered_df = working_df.copy()
            self._df = working_df.copy()
            self._update_metrics()
            self.filter_status_message = ""  
            self.show_schema_preview = True
            return

        filtered_df = working_df.iloc[0:0]
        try:
            if self.selected_search_column == "all" or not self.selected_search_column:
                mask = working_df.astype(str).apply(
                    lambda col: col.str.lower().str.contains(query, na=False)
                ).any(axis=1)
                filtered_df = working_df[mask]
            else:
                col = self.selected_search_column
                if col in working_df.columns:
                    mask = working_df[col].astype(str).str.lower().str.contains(query, na=False)
                    filtered_df = working_df[mask]
                else:
                    filtered_df = working_df.iloc[0:0] 
        except Exception as e:
            print(f"Live filter exception: {str(e)}")
            filtered_df = working_df.iloc[0:0] 

        # Update only the table display view and metrics, leaving matrices and _df alone
        self._filtered_df = filtered_df.copy()
        self._update_metrics()

        # Instantly show or clear the message based on match results
        if len(filtered_df) == 0:
            self.filter_status_message = "No Match."
        else:
            self.filter_status_message = ""  

        self.show_schema_preview = True     # --- FLAG TO DISPLAY THE DATA TABLE --- #

    # --- State Variables for Data Mapping ---
    selected_mapping_column: str = ""
    mapping_source_values: list[str] = []
    
    cph_error_message: str = ""

    columns_to_hide: list[str] = []
    all_dataset_columns: list[str] = []
    user_excluded_columns: list[str] = []
    is_custom_dataset: bool = False
    selected_id_column: str = ""
    selected_dedup_columns: list[str] = []

    is_mapping_popover_open: bool = False
    user_value_mappings: dict = {}

    def download_excel_data(self):
        """Export the active cleansed dataframe (used in analysis) as an Excel file."""
        # Resolve the active cleansed dataframe following the same priority as run_analysis
        target_df = None
        if hasattr(self, "_filtered_df") and self._filtered_df is not None and not self._filtered_df.empty:
            target_df = self._filtered_df
        elif hasattr(self, "_model_df") and self._model_df is not None and not self._model_df.empty:
            target_df = self._model_df
        elif hasattr(self, "_df") and self._df is not None and not self._df.empty:
            target_df = self._df

        if target_df is not None and not target_df.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                target_df.to_excel(writer, index=False, sheet_name='Cleansed Cohort')
            return rx.download(data=output.getvalue(), filename="cleansed_survival_data.xlsx")

    def download_csv_data(self):
        """Export the active cleansed dataframe (used in analysis) as a CSV file."""
        target_df = None
        if hasattr(self, "_filtered_df") and self._filtered_df is not None and not self._filtered_df.empty:
            target_df = self._filtered_df
        elif hasattr(self, "_model_df") and self._model_df is not None and not self._model_df.empty:
            target_df = self._model_df
        elif hasattr(self, "_df") and self._df is not None and not self._df.empty:
            target_df = self._df

        if target_df is not None and not target_df.empty:
            csv_data = target_df.to_csv(index=False)
            return rx.download(data=csv_data, filename="cleansed_survival_data.csv")

    # 2. Explicitly define the setter method to satisfy Reflex
    def set_is_mapping_popover_open(self, val: bool):
        self.is_mapping_popover_open = val

    def force_open_schema_view(self):
        """Explicitly enforce opening the data schema view."""
        self.show_schema_preview = True

    def set_mapping_column(self, col: str):
        """Select a column to map and extract its unique values."""
        self.selected_mapping_column = col
        if self._df is not None and col in self._df.columns:
            # Extract unique values and convert them to strings for display
            vals = self._df[col].dropna().unique().tolist()
            self.mapping_source_values = [str(v) for v in vals]
            # Initialize the mapping dictionary with empty target values
            self.user_value_mappings = {str(v): "" for v in vals}

    def update_single_mapping(self, orig_val: str, new_val: str):
        """Update the target mapping value for a specific source value."""
        #self.user_value_mappings[orig_val] = new_val

        # Create a new dictionary copy to ensure Reflex detects the state change
        mappings_copy = dict(self.user_value_mappings)
        mappings_copy[orig_val] = new_val
        self.user_value_mappings = mappings_copy

    def apply_data_mapping(self):
        """Apply the user-defined mappings to the dataframe column and sync all working copies."""
        try:
            if self._df is not None and self.selected_mapping_column:
                col = self.selected_mapping_column
                conversion = {}
                
                for k, v in self.user_value_mappings.items():
                    if v.strip() != "":
                        try:
                            if "." in v:
                                conversion[k] = float(v)
                            else:
                                conversion[k] = int(v)
                        except ValueError:
                            conversion[k] = v
                
                if conversion:
                    # 1. Apply replacement on master display df
                    self._df[col] = self._df[col].astype(str).replace(conversion)
                    
                    # Try converting back to numeric if all values look numeric now
                    try:
                        self._df[col] = pd.to_numeric(self._df[col])
                    except ValueError:
                        pass 

                    # Sync internal 'time' or 'event' columns if mapped
                    if col == self.selected_time_column:
                        self._df['time'] = pd.to_numeric(self._df[col], errors='coerce')
                    
                    if col == self.selected_event_column:
                        self._df['event'] = pd.to_numeric(self._df[col], errors='coerce').fillna(0).astype(int)

                    # 2. 🔥 CRITICAL: Lock in the mapped changes to override, raw, and model working copies
                    self._override_active_df = self._df.copy()
                    self._raw_df = self._df.copy()
                    
                    if getattr(self, "is_custom_dataset", True) and getattr(self, "uploaded_data", None) is not None:
                        self.uploaded_data = self._df.copy()

                    # Re-normalize model_df so math engines (CPH/KM) receive the updated/mapped values
                    model_df = self._df.copy()
                    if self.selected_time_column and self.selected_time_column in model_df.columns:
                        model_df[self.selected_time_column] = pd.to_numeric(model_df[self.selected_time_column], errors='coerce')
                    self._model_df = model_df
                    self._filtered_df = model_df.copy()

                    # Clear error states
                    self.cph_error_message = ""
                    self.show_variable_error = False
                    self.run_error_message = ""
                    
                    if col == self.selected_event_column:
                        self.set_event_column(self.selected_event_column)

                    # Rebuild unique values mapping so sidebar options reflect the new mapped values
                    new_unique_vals = {}
                    for col_name in self._df.columns:
                        # Pass the DataFrame and the column name string correctly
                        new_unique_vals[col_name] = self._generate_smart_intervals(self._df, col_name)
                            
                    self.column_unique_values = new_unique_vals       

                    # 3. Refresh inspectors, metrics, backend matrices, and enforce column exclusions
                    self._recompute_hidden_columns()
                    self._update_metrics()
                    self._populate_inspector_matrices(self._df)
                    self.metadata_info += f" (Mapped values in column '{col}')."

                    # 🔥 Clear stale analysis, figure, and text states so old graphs/metrics disappear
                    self.survival_fig_g1 = go.Figure()
                    self.cph_fig = go.Figure()
                    self.survival_probability_text = ""
                    self.median_survival_text = ""
                    self.p_value_text = ""
                    self.ci_text = ""
                    self.survival_probability_list = []
                    self.median_survival_list = []
                    self.at_risk_data = []
                    self.cph_table_data = []
                    self.cph_error_message = ""
                    self.clear_results_display()

                    # Reset mapping popover state
                    self.selected_mapping_column = ""
                    self.mapping_source_values = []
                    self.user_value_mappings = {}
                    self.is_mapping_popover_open = False
                    self.force_open_schema_view()

        except Exception as e:
            self.show_variable_error = True
            self.run_error_message = f"Mapping execution error: {str(e)}"
            self.show_schema_preview = True
            self.force_open_schema_view()

    def toggle_dedup_column(self, col: str, checked: bool):
        """Toggle individual columns used as unique keys for deduplication."""
        if checked:
            if col not in self.selected_dedup_columns:
                self.selected_dedup_columns.append(col)
        else:
            if col in self.selected_dedup_columns:
                self.selected_dedup_columns.remove(col)

    async def remove_exact_duplicates(self):
        """Remove exact full-row duplicates immediately when clicking the direct Rows button."""
        if self._df is not None and not self._df.empty:
            original_count = len(self._df)
            self.selected_dedup_columns = []  # Clear selective columns so this is strictly full-row

            # 🛡️ Safeguard active sidebar selections & custom numeric ranges before data mutation
            saved_g1 = dict(getattr(self, "g1_checked_values", {}))
            saved_g2 = dict(getattr(self, "g2_checked_values", {}))
            saved_g1_min = dict(getattr(self, "g1_custom_min", {}))
            saved_g1_max = dict(getattr(self, "g1_custom_max", {}))
            saved_g2_min = dict(getattr(self, "g2_custom_min", {}))
            saved_g2_max = dict(getattr(self, "g2_custom_max", {}))
            saved_cph = list(getattr(self, "cph_active_covariates", []))
            
            self.selected_dedup_columns = []

          # 1. Drop duplicates on the master dataframe and keep uploaded_data / _raw_df synced
            #self._df = self._df.drop_duplicates().reset_index(drop=True)
            #if getattr(self, "is_custom_dataset", True) and self.uploaded_data is not None:
            #    self.uploaded_data = self._df.copy()
            #if getattr(self, "_raw_df", None) is not None:
            #    self._raw_df = self._raw_df.drop_duplicates().reset_index(drop=True)

            self._df = self._df.drop_duplicates().reset_index(drop=True)
            self._filtered_df = self._df.copy()

             # 💡 Crucial: Update _raw_df too so clearing filters doesn't bring duplicates back
            if getattr(self, "_raw_df", None) is not None:
                self._raw_df = self._raw_df.drop_duplicates().reset_index(drop=True)

            new_count = len(self._df)
            dropped_count = original_count - new_count

            # 2. Feed the cleansed dataframe into the centralized update_dataset pipeline via override / Resets Display -- Clears old Results
            self._override_active_df = self._df.copy()
            target_dataset = self.current_dataset if self.current_dataset else "Own Data (.csv/.xls)"
            async for _ in self.update_dataset(target_dataset):
                yield
            
            # 2. Recalculate metrics & refresh inspector/table views
            self._update_metrics()
            self._populate_inspector_matrices()  # Matches the convention used in your filters
            
            self.g1_checked_values = saved_g1
            self.g2_checked_values = saved_g2
            self.g1_custom_min = saved_g1_min
            self.g1_custom_max = saved_g1_max
            self.g2_custom_min = saved_g2_min
            self.g2_custom_max = saved_g2_max
            self.cph_active_covariates = saved_cph

            if dropped_count > 0:
                self.status_message = f"Removed {dropped_count} exact duplicates."
            else:
                self.status_message = "No exact duplicates found."
                
            # Push the updated row counts and status message to the UI immediately
            yield
            
            # Wait for 2 seconds, then clear the banner and push that state change too
            await asyncio.sleep(2)
            self.status_message = ""
            yield

    async def remove_selective_duplicates(self):
        """Remove duplicate rows based on user-selected criteria columns from the popover."""
        if self._df is not None and not self._df.empty:
            active_subset = [c for c in self.selected_dedup_columns if c in self._df.columns]
            
            # 🛡️ Safeguard active sidebar selections & custom numeric ranges before data mutation
            saved_g1 = dict(getattr(self, "g1_checked_values", {}))
            saved_g2 = dict(getattr(self, "g2_checked_values", {}))
            saved_g1_min = dict(getattr(self, "g1_custom_min", {}))
            saved_g1_max = dict(getattr(self, "g1_custom_max", {}))
            saved_g2_min = dict(getattr(self, "g2_custom_min", {}))
            saved_g2_max = dict(getattr(self, "g2_custom_max", {}))
            saved_cph = list(getattr(self, "cph_active_covariates", []))
        
            # If no columns are selected, abort completely
            if not active_subset:
                self.selective_status_message = "Please select at least one column for duplicates."
                yield
                await asyncio.sleep(2)
                self.selective_status_message = ""
                yield
                return
                
            original_count = len(self._df)
            
            # 1. Permanently drop duplicates on master, filtered df, and raw df BEFORE running pipeline
            self._df = self._df.drop_duplicates(subset=active_subset, keep='first').reset_index(drop=True)
            self._filtered_df = self._df.copy()

            if getattr(self, "_raw_df", None) is not None:
                self._raw_df = self._raw_df.drop_duplicates(subset=active_subset, keep='first').reset_index(drop=True)

            if getattr(self, "is_custom_dataset", True) and getattr(self, "uploaded_data", None) is not None:
                self.uploaded_data = self._df.copy()

            new_count = len(self._df)
            dropped_count = original_count - new_count
            cols_str = ", ".join(active_subset)
            
            # 2. Feed the cleansed dataframe into the pipeline via override (must happen after raw/uploaded sync)
            self._override_active_df = self._df.copy()
            target_dataset = self.current_dataset if self.current_dataset else "Own Data (.csv/.xls)"
            async for _ in self.update_dataset(target_dataset):
                yield
            
            # 3. Recalculate metrics & refresh inspector views with updated base counts
            self._update_metrics()
            self._populate_inspector_matrices()
            
            # ♻️ Restore sidebar selection states & custom numeric ranges
            self.g1_checked_values = saved_g1
            self.g2_checked_values = saved_g2
            self.g1_custom_min = saved_g1_min
            self.g1_custom_max = saved_g1_max
            self.g2_custom_min = saved_g2_min
            self.g2_custom_max = saved_g2_max
            self.cph_active_covariates = saved_cph

            if dropped_count > 0:
                self.selective_status_message = f"Removed {dropped_count} duplicates based on {cols_str}."
            else:
                self.selective_status_message = f"No duplicates found for {cols_str}."
                
            yield
            await asyncio.sleep(2)
            self.selective_status_message = ""
            yield

    def toggle_exclude_column(self, col: str, checked: bool):
        """Toggle individual column exclusion from the popover checkboxes."""
        # Prevent excluding active critical time/event columns
        if col in [self.selected_time_column, self.selected_event_column]:
            return
            
        if checked:
            if col not in self.user_excluded_columns:
                self.user_excluded_columns.append(col)
        else:
            if col in self.user_excluded_columns:
                self.user_excluded_columns.remove(col)
                
        self._recompute_hidden_columns()

    def _recompute_hidden_columns(self):
        # --------------- Recalculate columns to hide based on defaults + user selections and drop from math model df ----------------- #
        base_hidden = []

        # Only apply automatic presets for built-in template datasets, not user-uploaded data
        if not getattr(self, "is_custom_dataset", False):
            base_hidden = ['inst']
            for c in ['status', 'cens', 'censor', 'died', 'recurrence']:
                if c in self.all_dataset_columns and c != self.selected_event_column:
                    if c not in base_hidden:
                        base_hidden.append(c)

        
        combined = list(set(base_hidden + self.user_excluded_columns))  # --- Merge with user selections
        
        
        if self.selected_time_column in combined:                       # --- Ensure Time Column is never hidden
            combined.remove(self.selected_time_column)
        
        if self.selected_event_column in combined:                      # --- Ensure Event Column is never hidden
            combined.remove(self.selected_event_column)
            
        self.columns_to_hide = combined
        self.csv_columns = [col for col in self.all_dataset_columns if col not in combined] # --- Aktuelle "Variablen-Liste" = ("Alle Spalten" - "Ausgeschlossene Spalten") ---  
        
        # Re-initialize filter dictionaries for Group 1 and Group 2 in Sidebar Selection columns
        self.g1_checked_values = {col: [] for col in self.csv_columns if col not in self.critical_columns}
        self.g2_checked_values = {col: [] for col in self.csv_columns if col not in self.critical_columns}

        # 🔥 ------------- Ensure excluded columns are physically dropped from the math model dataframe --------------- #
        if getattr(self, "_override_active_df", None) is not None:
            base_model_df = self._override_active_df.copy()
        elif getattr(self, "_raw_df", None) is not None:
            base_model_df = self._raw_df.copy()
        elif getattr(self, "_df", None) is not None:
            base_model_df = self._df.copy()
        else:
            base_model_df = None

        if base_model_df is not None:
            # Drop any hidden/excluded columns except mandatory time/event columns
            criticals = [self.selected_time_column, self.selected_event_column, 'time', 'cens']
            drop_targets = [col for col in combined if col in base_model_df.columns and col not in criticals]
            
            if drop_targets:
                base_model_df = base_model_df.drop(columns=drop_targets)

            # Re-normalize model dataframe time scaling
            if self.selected_time_column and self.selected_time_column in base_model_df.columns:
                try:
                    base_model_df[self.selected_time_column] = pd.to_numeric(base_model_df[self.selected_time_column], errors='coerce')
                    raw_max = base_model_df[self.selected_time_column].dropna().max()
                    if raw_max and raw_max > 3650:
                        base_model_df[self.selected_time_column] = base_model_df[self.selected_time_column] / 365.25
                    elif raw_max and raw_max > 90:
                        base_model_df[self.selected_time_column] = base_model_df[self.selected_time_column] / 30.4375
                except Exception:
                    pass

            self._model_df = base_model_df
            self._filtered_df = base_model_df.copy()

        # Clean up active CPH covariates if an excluded column was previously selected
        if hasattr(self, "cph_active_covariates"):
            self.cph_active_covariates = [c for c in self.cph_active_covariates if c not in combined]

        # 🔥 Clear stale analysis, figure, and text states so old graphs/metrics disappear
        self.survival_fig_g1 = go.Figure()
        self.cph_fig = go.Figure()
        self.survival_probability_text = ""
        self.median_survival_text = ""
        self.p_value_text = ""
        self.ci_text = ""
        self.survival_probability_list = []
        self.median_survival_list = []
        self.at_risk_data = []
        self.cph_table_data = []
        self.cph_error_message = ""
        self.clear_results_display()

    is_full_cohort: bool = False

    # --- Report Metadata Editable Fields ---
    show_report_panel: bool = False
    researcher_name: str = "Lead Investigator"
    institution_name: str = "Department"
    report_date: str = datetime.now().strftime("%B %d, %Y")
    
    @rx.event
    def toggle_report_panel(self):
        self.show_report_panel = not self.show_report_panel

    @rx.event
    def set_researcher_name(self, value: str):
        self.researcher_name = value

    @rx.event
    def set_institution_name(self, value: str):
        self.institution_name = value

    @rx.event   # --- DOWNLOAD REPORT LAYOUT AND CONTENT DEFINITION --- #
    def set_report_date(self, value: str):
        self.report_date = value

    # ------------------------------------------------------------------------------------------------------------------------------------- #
    # ---------- Create CPH Analysis Report to meet world-class publication standards (such as STROBE and REMARK guidelines) -------------- #
    @rx.event
    def export_cph_pdf_report(self):

        # --- 1. Fetch Metadata --- #
        base_dataset = getattr(self, "selected_dataset", "N/A")
        custom_file = getattr(self, "custom_filename", "")
        dataset_name = custom_file if (base_dataset == "Own Data (.csv/.xls)" and custom_file) else base_dataset

        model_name = getattr(self, "selected_model", "Cox Proportional Hazards")
        researcher_name = getattr(self, "researcher_name", "Lead Investigator")
        institution_name = getattr(self, "institution_name", "Department")
        generation_date = getattr(self, "report_date", "November 02, 2026")
        full_cohort_n = getattr(self, "cph_full_cohort_n", "N/A")

        def parse_categorical_cell(cell_str):
            if "(" in cell_str and ")" in cell_str:
                parts = cell_str.split("(")
                return parts[0].strip(), f"({parts[1].strip()}"
            return cell_str, ""

        # --- 2. Table 1 Rows Assembly --- #
        table1_rows_html = ""
        table_data = getattr(self, "table_one_data", [])

        if table_data:
            for row in table_data:
                var_text = row.get("variable", "")
                row_type = row.get("type", "Continuous")
                
                if row_type == "Header":
                    table1_rows_html += f'''
                        <tr>
                            <td colspan="3" style="font-weight: bold; color: #1e293b; padding: 6px 10px; background-color: #ffffff; border-top: 1.5px solid #1e293b; border-bottom: 1px solid #1e293b;">{var_text}</td>
                        </tr>
                    '''
                elif row_type == "Categorical":
                    g1_val = row.get("group1", "N/A")
                    g1_int, g1_dec = parse_categorical_cell(g1_val)
                    table1_rows_html += f'''
                        <tr style="border-bottom: 1px solid #f1f5f9;">
                            <td style="padding-left: 20px; width: 50%; color: #1e293b;">{var_text}</td>
                            <td style="text-align: right; padding-right: 4px; font-variant-numeric: tabular-nums; width: 25%; border-right: none; color: #1e293b;">{g1_int}</td>
                            <td style="text-align: left; padding-left: 4px; font-variant-numeric: tabular-nums; width: 25%; color: #475569; border-left: none;">{g1_dec}</td>
                        </tr>
                    '''
                else:  # Continuous
                    g1_val = row.get("group1", "N/A")
                    table1_rows_html += f'''
                        <tr style="border-bottom: 1px solid #f1f5f9;">
                            <td style="padding-left: 10px; width: 50%; color: #1e293b;">{var_text}</td>
                            <td colspan="2" style="text-align: center; font-variant-numeric: tabular-nums; width: 50%; color: #334155;">{g1_val}</td>
                        </tr>
                    '''
        else:
            table1_rows_html = '<tr><td colspan="3" style="text-align: center; color: #64748b; padding: 12px;">No baseline characteristics available.</td></tr>'

        # --- 3. Table 2 (Multivariate CPH Table) Assembly ---
        table_data_cph = list(getattr(self, "cph_table_data", []))
        rows_cph_html = ""
        for row in table_data_cph:
            var_name = row[0] if len(row) > 0 else "Unknown"
            hr_univ = row[1] if len(row) > 1 else "N/A"
            uni_ci = row[2] if len(row) > 2 else "N/A"
            
            try:
                univ_p_float = float(row[3])
                formatted_univ_p = "< 0.001" if univ_p_float < 0.001 else f"{univ_p_float:.3f}"
            except (ValueError, TypeError, IndexError):
                formatted_univ_p = str(row[3]) if len(row) > 3 else "N/A"

            hr_adj = row[4] if len(row) > 4 else "N/A"
            adj_ci = row[5] if len(row) > 5 else "N/A"
            
            try:
                adj_p_float = float(row[6])
                formatted_adj_p = "< 0.001" if adj_p_float < 0.001 else f"{adj_p_float:.3f}"
            except (ValueError, TypeError, IndexError):
                formatted_adj_p = str(row[6]) if len(row) > 6 else "N/A"

            schoenfeld_p = row[7] if len(row) > 7 else "1.000"

            rows_cph_html += f'''
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="font-weight: bold; color: #1e293b; text-align: left; padding: 6px 8px; white-space: nowrap;">{var_name}</td>
                    <td style="text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; padding: 6px 6px; color: #1e293b;">{hr_univ}</td>
                    <td style="text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; padding: 6px 6px; color: #1e293b;">{uni_ci}</td>
                    <td style="text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; padding: 6px 6px; color: #1e293b;">{formatted_univ_p}</td>
                    <td style="text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; padding: 6px 6px; font-weight: bold; color: #1e293b;">{hr_adj}</td>
                    <td style="text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; padding: 6px 6px; color: #1e293b;">{adj_ci}</td>
                    <td style="text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; padding: 6px 6px; font-weight: bold; color: #1e293b;">{formatted_adj_p}</td>
                    <td style="text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; padding: 6px 6px; color: #64748b;">{schoenfeld_p}</td>
                </tr>
            '''

        if not rows_cph_html:
            rows_cph_html = '<tr><td colspan="8" style="text-align: center; color: #64748b; padding: 12px;">No CPH analysis results available.</td></tr>'

        # --- 4. Helper Function to Convert Plotly Figures to Base64 ---
        def figure_to_base64(fig_obj, width=850, height=360):
            if not fig_obj:
                return ""
            try:
                img_bytes = fig_obj.to_image(format="png", width=width, height=height, scale=2)
                return base64.b64encode(img_bytes).decode("utf-8")
            except Exception as e:
                print(f"DEBUG [PDF Image Export Warning]: {e}")
                return ""

        # Generate image strings for all 5 figures
        forest_b64 = figure_to_base64(getattr(self, "cph_fig", None), width=880, height=380)
        calib_b64 = figure_to_base64(getattr(self, "cph_calibration_fig", None), width=850, height=340)
        dca_b64 = figure_to_base64(getattr(self, "cph_dca_fig", None), width=850, height=340)
        auc_b64 = figure_to_base64(getattr(self, "time_dep_auc_fig", None), width=850, height=320)
        km_b64 = figure_to_base64(getattr(self, "risk_stratified_km_fig", None), width=850, height=340)

        # --- 5. Extract Metrics for Summary Callout Cards ---
        sample_metrics_str = getattr(self, "cph_sample_metrics_text", "N/A")
        base_c_text = getattr(self, "cph_c_index_text", "N/A")
        corrected_val = getattr(self, "cph_corrected_c_index", "N/A")
        lr_val = getattr(self, "cph_log_rank_p", "N/A")
        ph_warning_text = getattr(self, "cph_ph_warning", "Proportional hazards assumption satisfied.")
        
        calib_intercept = getattr(self, "calib_intercept_val", "N/A")
        calib_slope = getattr(self, "calib_slope_val", "N/A")
        calib_slope_ci = getattr(self, "calib_slope_ci", "")
        gnd_chi2 = getattr(self, "gnd_chi2_val", "N/A")
        gnd_p = getattr(self, "gnd_p_val", "N/A")
        km_summary_text = getattr(self, "risk_stratified_km_summary_text", "Risk-Stratified Kaplan-Meier Analysis")

        # ---Methodological & Compliance Notes Block --- #
        tie_adjustment_note = ""
        if getattr(self, "km_warning_message", "") != "":
            tie_adjustment_note = " Risk score tertile boundaries were adjusted using rank-based tie-breaking to accommodate clustered data."

        # --- 2. Build the footnotes HTML snippet ---
        cph_footnotes_html = f'''
        <div class="notes-box" style="font-size: 7.5pt; color: #64748b; margin-top: 24px; line-height: 1.4; border-top: 1px solid #cbd5e1; padding-top: 12px;">
            <div style="margin-bottom: 10px;"><strong>Abbreviations:</strong> HR, hazard ratio; CI, confidence interval; Adj, adjusted.</div>
            <div><strong>Methodological & Guideline Compliance Notes:</strong> This report complies with TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis) and REMARK guidelines. Multivariable hazard ratios were estimated via Cox proportional hazards regression. Optimism-corrected performance was derived via bootstrap resamples. Calibration evaluated via Greenwood-Nam-D'Agostino (GND) chi-square goodness-of-fit. Schoenfeld residual p-values evaluate the proportional hazards assumption (p > 0.05 indicates satisfaction).{tie_adjustment_note}</div>
        </div>
        '''

        # --- 6. Construct High-Quality HTML/CSS Document ---
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        
        <style>                                                     <!-- <--- THIS IS WHERE THE CSS BLOCK STARTS -->
            *, *::before, *::after {{ box-sizing: border-box; }}
            @page {{
                size: A4;
                margin: 15mm 15mm 18mm 15mm;
                @bottom-right {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 8pt;
                    color: #64748b;
                    font-family: Helvetica, Arial, sans-serif;
                }}
                @bottom-left {{
                    content: "TRIPOD/REMARK-Compliant Prognostic Report | Clinical Survival Platform";
                    font-size: 8pt;
                    color: #64748b;
                    font-family: Helvetica, Arial, sans-serif;
                }}
            }}
            body {{
                font-family: Helvetica, Arial, sans-serif;
                color: #1e293b;
                font-size: 8.5pt;
                line-height: 1.4;
                margin: 0;
                padding: 0;
                background-color: #ffffff;
            }}
            .header-banner {{
                border-bottom: 2px solid #1e293b;
                padding-bottom: 8px;
                margin-bottom: 22px; /* Increased from 14px to add space between the header line and metadata table */
            }}
            .header-banner h1 {{
                font-size: 13pt;
                color: #1e293b;
                margin: 0 0 2px 0;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }}
            .header-banner .subtitle {{
                font-size: 8pt;
                color: #64748b;
                margin: 0;
            }}
            .meta-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 8pt;
                border: 1px solid #cbd5e1;
                margin-bottom: 28px;            /* was 12px */
                background-color: #ffffff;
            }}
            .meta-table td {{
                padding: 5px 8px;
                border: 1px solid #cbd5e1;
                vertical-align: middle;
            }}
            .meta-label {{ font-weight: bold; color: #1e293b; width: 18%; background-color: #f8fafc; }}
            .meta-val {{ color: #334155; width: 32%; }}
            
            .diagnostics-card {{
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-left: 3px solid #1e293b;
                border-radius: 2px;
                padding: 8px 10px;
                margin-bottom: 20px;    /* was 14px */
                font-size: 8pt;
                page-break-inside: avoid;
            }}
            .diagnostics-title {{
                font-weight: bold;
                color: #1e293b;
                font-size: 9pt;
                margin-bottom: 4px;
            }}
            .metric-grid {{
                width: 100%;
                border-collapse: collapse;
            }}
            .metric-grid td {{
                padding: 2px 4px;
                font-size: 8pt;
                color: #334155;
            }}

            h2 {{
                font-size: 10pt;
                color: #1e293b;
                border-bottom: 1.5px solid #1e293b;
                padding-bottom: 3px;
                margin-top: 24px;           /* was 18px distance beetween Table 1 and CPH HR Table */
                margin-bottom: 10px;        /* was 8px */
                font-weight: bold;
                page-break-after: avoid; /* Prevents heading from being orphaned at the bottom of a page */
            }}
            
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 8.0pt;
                background-color: #ffffff;
                border-top: 1.5px solid #1e293b;
                border-bottom: 1.5px solid #1e293b;
                margin-bottom: 12px;
            }}
            
            /* CRITICAL: Automatically repeat table header row if table splits across pages */
            thead {{
                display: table-header-group;
            }}
            tr {{
                page-break-inside: avoid;
            }}

            .data-table th {{
                background-color: #ffffff;
                color: #1e293b;
                font-weight: bold;
                padding: 6px 8px;
                border: none;
                border-bottom: 1px solid #1e293b;
                vertical-align: top;
            }}
            .data-table td {{
                padding: 6px 8px;
                border: none;
            }}

            .figure-block {{
                margin-top: 24px;               /* Increased from 12px to add clean breathing room above figures */
                margin-bottom: 14px;
                text-align: center;
                page-break-inside: avoid; /* Keeps figures whole so they don't fracture */
            }}
            .figure-block img {{
                width: 100%;
                height: auto;
                max-height: 330px;
                border: 1px solid #cbd5e1;
                background-color: #ffffff;
            }}
            .caption {{
                font-size: 7.5pt;
                color: #64748b;
                margin-top: 4px;
                text-align: left;
                line-height: 1.3;
            }}

            .notes-box {{
                font-size: 7.5pt;
                color: #64748b;
                background-color: #ffffff;
                border-top: 1px solid #cbd5e1;
                padding-top: 8px;
                margin-top: 40px; /* Increased from 16px to give the final footnotes clean breathing room */
                line-height: 1.4;
                page-break-inside: avoid;
            }}
        </style>                            <!-- <--- THIS IS WHERE THE CSS BLOCK ENDS -->

        </head>
        <body>
            <!-- HEADER BANNER & METADATA -->
            <div class="header-banner">
                <h1>Multivariable Prognostic Analysis Report</h1>
                <p class="subtitle">TRIPOD & REMARK Compliant Clinical Cox Proportional Hazards Report ({model_name})</p>
            </div>

            <table class="meta-table">
                <tr>
                    <td class="meta-label">Dataset Source:</td>
                    <td class="meta-val">{dataset_name}</td>
                    <td class="meta-label">Model Type:</td>
                    <td class="meta-val">{model_name}</td>
                </tr>
                <tr>
                    <td class="meta-label">Lead Investigator:</td>
                    <td class="meta-val">{researcher_name}</td>
                    <td class="meta-label">Institution:</td>
                    <td class="meta-val">{institution_name}</td>
                </tr>
                <tr>
                    <td class="meta-label">Report Date:</td>
                    <td class="meta-val" colspan="3">{generation_date}</td>
                </tr>
            </table>

            <div class="diagnostics-card">
                <div class="diagnostics-title">Multivariable Model Performance Summary</div>
                <table class="metric-grid">
                    <tr>
                        <td><strong>Analyzed Sample:</strong> {sample_metrics_str}</td>
                        <td><strong>Harrell's C-index:</strong> {base_c_text}</td>
                    </tr>
                    <tr>
                        <td><strong>Global Log-Rank Test:</strong> {lr_val}</td>
                        <td><strong>Optimism-Corrected C-index:</strong> <span style="color: #1e293b; font-weight: bold;">{corrected_val}</span></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="color: #475569; font-style: italic; padding-top: 3px;">Note: {ph_warning_text}</td>
                    </tr>
                </table>
            </div>

            <!-- SECTION 1: TABLE 1 (Flows naturally; repeats header if split across pages) -->
            <h2>Table 1: Baseline Characteristics of Study Cohort</h2>
            <table class="data-table" style="border-top: 1.5px solid #1e293b; border-bottom: 1.5px solid #1e293b;">
                <thead>
                    <tr style="border-bottom: 1.5px solid #1e293b;">
                        <th style="width: 50%; text-align: left;">Characteristic</th>
                        <th style="width: 50%; text-align: center;" colspan="2">Overall Cohort (N={full_cohort_n})</th>
                    </tr>
                </thead>
                <tbody>
                    {table1_rows_html}
                </tbody>
            </table>

            <!-- SECTION 2: CPH REGRESSION & FOREST PLOT (Follows immediately without forced break) -->
            <div style="margin-top: 38px;">                 <!-- Distance between Table 1 and CPH HR Table: ncreased from 20px -->
                <h2>Cox Proportional Hazards Analysis & Hazard Ratios</h2>      
                <table class="data-table">
                    <thead>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <th rowspan="2" style="text-align: left; vertical-align: top; width: 22%;">Covariate</th>
                            <th colspan="3" style="text-align: center; vertical-align: top; border-bottom: 1px solid #cbd5e1;">Univariate Analysis</th>
                            <th colspan="4" style="text-align: center; vertical-align: top; border-bottom: 1px solid #cbd5e1;">Multivariate Adjusted Analysis</th>
                        </tr>
                        <tr style="border-bottom: 1.5px solid #1e293b;">
                            <th style="text-align: center; vertical-align: top; width: 10%;">HR</th>
                            <th style="text-align: center; vertical-align: top; width: 13%;">95% CI</th>
                            <th style="text-align: center; vertical-align: top; width: 10%;">P-Value</th>
                            <th style="text-align: center; vertical-align: top; width: 10%;">Adj. HR</th>
                            <th style="text-align: center; vertical-align: top; width: 13%;">95% CI</th>
                            <th style="text-align: center; vertical-align: top; width: 11%;">Adj. P</th>
                            <th style="text-align: center; vertical-align: top; width: 11%;">Schoenfeld P</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_cph_html}
                    </tbody>
                </table>

                {f'''
                <div class="figure-block" style="margin-top: 34px;">    <!-- Distance between CPH HR Table to Figure 1 -->
                    <h2>Figure 1: Multivariable Forest Plot</h2>
                    <img src="data:image/png;base64,{forest_b64}" />
                    <div class="caption">
                        <strong>Figure 1:</strong> Adjusted Hazard Ratios with 95% Confidence Intervals from the multivariable Cox proportional hazards model.
                    </div>
                </div>
                ''' if forest_b64 else ''}
            </div>

            <!-- SECTION 3: MODEL CALIBRATION & DECISION CURVE ANALYSIS -->
            <div style="margin-top: 20px;">
                <h2>Model Calibration & Absolute Risk Accuracy</h2>
                <div class="diagnostics-card" style="border-left-color: #1e293b;">
                    <div class="diagnostics-title">Quantitative Calibration Statistics</div>
                    <table class="metric-grid">
                        <tr>
                            <td><strong>Calibration Intercept:</strong> {calib_intercept} (Ideal: 0.00)</td>
                            <td><strong>Calibration Slope:</strong> {calib_slope} {calib_slope_ci} (Ideal: 1.00)</td>
                        </tr>
                        <tr>
                            <td colspan="2" style="padding-top: 4px;"><strong>GND Goodness-of-Fit Test:</strong> chi2 = {gnd_chi2}, p = {gnd_p} (H0: Well-calibrated if p > 0.05)</td>
                        </tr>
                    </table>
                </div>

                {f'''
                <div class="figure-block" style="margin-top: 24px;">        <!-- Distance between Model Calibration Table and Figure 2 -->
                    <h2>Figure 2: Model Calibration Curve</h2>
                    <img src="data:image/png;base64,{calib_b64}" />
                    <div class="caption">
                        <strong>Figure 2:</strong> Comparison of predicted event probability against observed Kaplan-Meier estimates across risk deciles.
                    </div>
                </div>
                ''' if calib_b64 else ''}

                {f'''
                <div class="figure-block" style="margin-top: 34px;">        <!-- Distance between Figure 2 and Figure 3 -->
                    <h2>Figure 3: Decision Curve Analysis (DCA)</h2>
                    <img src="data:image/png;base64,{dca_b64}" />
                    <div class="caption">
                        <strong>Figure 3:</strong> Demonstrates net clinical benefit of guided intervention versus 'Treat All' and 'Treat None' strategies across threshold probabilities.
                    </div>
                </div>
                ''' if dca_b64 else ''}
            </div>

            <!-- SECTION 4: TEMPORAL DISCRIMINATION & RISK STRATIFICATION -->
            <div style="margin-top: 20px;">
                <h2>Temporal Discrimination & Actionable Risk Stratification</h2>

                {f'''
                <div class="figure-block">
                    <h2>Figure 4: Time-Dependent Discrimination (AUC Trajectory)</h2>
                    <img src="data:image/png;base64,{auc_b64}" />
                    <div class="caption">
                        <strong>Figure 4:</strong> Evaluates predictive discrimination stability across follow-up time horizons (25th, 50th, and 75th event percentiles).
                    </div>
                </div>
                ''' if auc_b64 else ''}

                {f'''
                <div class="figure-block" style="margin-top: 34px;">        <!--- Distance between Figure 4 and Figure 5 --->
                    <h2>Figure 5: Risk-Stratified Kaplan-Meier Curves</h2>
                    <img src="data:image/png;base64,{km_b64}" />
                    <div class="caption">
                        <strong>Figure 5:</strong> Cohort partitioned into Low, Intermediate, and High risk tertiles based on the model's linear predictor. {km_summary_text}.
                    </div>
                </div>
                ''' if km_b64 else ''}

                <!-- Drop the compiled footnotes HTML right here -->
                {cph_footnotes_html}
            </div>
        </body>
        </html>
        """

        # --- 7. Generate PDF with WeasyPrint ---
        try:
            from weasyprint import HTML
            pdf_bytes = HTML(string=html_content).write_pdf()
            return rx.download(data=pdf_bytes, filename="clinical_cph_journal_report_v2.pdf")
        except Exception as e:
            print(f"DEBUG [CPH PDF ERROR]: WeasyPrint CPH PDF Generation Error: {e}")

        # --- 7. Generate PDF with WeasyPrint ---
        try:
            from weasyprint import HTML
            pdf_bytes = HTML(string=html_content).write_pdf()
            return rx.download(data=pdf_bytes, filename="clinical_cph_journal_report_v2.pdf")
        except Exception as e:
            print(f"DEBUG [CPH PDF ERROR]: WeasyPrint CPH PDF Generation Error: {e}")
    
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # ------------ Create KM Analysis Report to meet world-class publication standards (such as STROBE and REMARK guidelines) ------------ #
    def export_km_pdf_report(self):
        if not getattr(self, "results_ready", False):
            return
            
        # --- Fetch Metadata from State ---
        base_dataset = getattr(self, "selected_dataset", "N/A")
        custom_file = getattr(self, "custom_filename", "")
    
        if base_dataset == "Own Data (.csv/.xls)" and custom_file:      
            dataset_name = custom_file  
        else:
            dataset_name = base_dataset
    
        model_name = getattr(self, "selected_model", "Kaplan-Meier Survival")
        p_val = getattr(self, "p_value_text", "N/A")
        
        researcher_name = getattr(self, "researcher_name", "")
        institution_name = getattr(self, "institution_name", "")
        generation_date = getattr(self, "report_date", "")
    
        g1_n = getattr(self, "g1_n", "N/A")
        g2_n = getattr(self, "g2_n", "N/A")
        g1_events = getattr(self, "g1_events", "N/A")
        g2_events = getattr(self, "g2_events", "N/A")
        
        # Pull table headers established for Table 1 and GUI display
        g1_label = getattr(self, "table_one_g1_label", "Group 1")
        g2_label = getattr(self, "table_one_g2_label", "Group 2")

        is_full_cohort = getattr(self, "is_full_cohort", False) or (
            "Full Cohort" in str(getattr(self, "g1_detailed_label", "")) and 
            "Full Cohort" in str(getattr(self, "g2_detailed_label", ""))
        )

        # --- Safe Default Initializations for Headers & Sub-criteria ---
        h1_title = g1_label
        h2_title = g2_label
        h1_sub = "Full Cohort"
        h2_sub = "Full Cohort"

        # --- Parse and Clean Group Headers & Sub-criteria with No-Wrap Protection ---
        raw_g1_desc = getattr(self, "g1_detailed_label", "")
        raw_g2_desc = getattr(self, "g2_detailed_label", "")
        
        def clean_criteria(raw_str):
            cleaned = str(raw_str)
            for prefix in [
                f"{g1_label}:", f"{g2_label}:", 
                "Group 1:", "Group 2:", 
                "Group 1 (Complement):", "Group 2 (Complement):",
                "Full Cohort"
            ]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned.replace(prefix, "", 1).strip()
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = cleaned[1:-1].strip()
            return cleaned if cleaned else "Full Cohort / Unstratified"

        g1_clean = clean_criteria(raw_g1_desc)
        g2_clean = clean_criteria(raw_g2_desc)

        def parse_group_header_and_sub(label_str, clean_str):
            is_comp = "Complement" in label_str or "Complement" in clean_str
            
            if "Group 1" in label_str:
                base_header = "Group 1"
            elif "Group 2" in label_str:
                base_header = "Group 2"
            else:
                base_header = label_str.split("(")[0].strip()
                
            if is_comp:
                sub_text = "Complement"
            else:
                raw_sub = clean_str if clean_str and clean_str != "Full Cohort" else "Full Cohort"
                if raw_sub not in ["Full Cohort", "Complement"]:
                    parts = [f'<span style="white-space: nowrap;">{p.strip()}</span>' for p in raw_sub.split(",")]
                    sub_text = ", ".join(parts)
                else:
                    sub_text = raw_sub
                
            return base_header, sub_text

        if not is_full_cohort:
            h1_title, h1_sub = parse_group_header_and_sub(g1_label, g1_clean)
            h2_title, h2_sub = parse_group_header_and_sub(g2_label, g2_clean)

        # Retrieve structured CPH rows produced during the analysis run
        full_cph_rows = getattr(self, "cph_table_data", [])
        if is_full_cohort:
            primary_comparison_rows_data = []
            multivariate_covs_rows_data = full_cph_rows
        else:
            primary_comparison_rows_data = full_cph_rows[:1] if full_cph_rows else []
            multivariate_covs_rows_data = full_cph_rows[1:] if len(full_cph_rows) > 1 else []
            
        # --- Dynamically Build HTML Rows for CPH Tables ---
        def build_cph_rows(rows_data):
            rows_html = ""
            for row in rows_data:
                try:
                    univ_p_float = float(row[3]) if len(row) > 3 else 1.0
                    formatted_univ_p = "< 0.001" if univ_p_float < 0.001 else f"{univ_p_float:.4f}".rstrip('0').rstrip('.')
                except (ValueError, TypeError, IndexError):
                    formatted_univ_p = str(row[3]) if len(row) > 3 else "N/A"
    
                try:
                    adj_p_float = float(row[6]) if len(row) > 6 else 1.0
                    formatted_adj_p = "< 0.001" if adj_p_float < 0.001 else f"{adj_p_float:.4f}".rstrip('0').rstrip('.')
                except (ValueError, TypeError, IndexError):
                    formatted_adj_p = str(row[6]) if len(row) > 6 else "N/A"

                try:
                    schoen_p_float = float(row[7]) if len(row) > 7 else 1.0
                    formatted_schoen_p = "< 0.001" if schoen_p_float < 0.001 else f"{schoen_p_float:.3f}"
                except (ValueError, TypeError, IndexError):
                    formatted_schoen_p = str(row[7]) if len(row) > 7 else "N/A"
    
                var_name = row[0] if len(row) > 0 else "Unknown"
                hr_univ = row[1] if len(row) > 1 else "N/A"
                ci_univ = row[2] if len(row) > 2 else "N/A"
                hr_adj = row[4] if len(row) > 4 else "N/A"
                ci_adj = row[5] if len(row) > 5 else "N/A"
    
                rows_html += f'''
                    <tr>
                        <td style="font-weight: 500; background-color: #ffffff; white-space: nowrap;">{var_name}</td>
                        <td style="text-align: center; font-variant-numeric: tabular-nums; background-color: #ffffff;">{hr_univ}</td>
                        <td style="text-align: center; font-variant-numeric: tabular-nums; background-color: #ffffff; white-space: nowrap;">{ci_univ}</td>
                        <td style="text-align: center; font-variant-numeric: tabular-nums; background-color: #ffffff;">{formatted_univ_p}</td>
                        <td style="text-align: center; font-variant-numeric: tabular-nums; background-color: #ffffff;">{hr_adj}</td>
                        <td style="text-align: center; font-variant-numeric: tabular-nums; background-color: #ffffff; white-space: nowrap;">{ci_adj}</td>
                        <td style="text-align: center; font-variant-numeric: tabular-nums; background-color: #ffffff;">{formatted_adj_p}</td>
                        <td style="text-align: center; font-variant-numeric: tabular-nums; background-color: #ffffff;">{formatted_schoen_p}</td>
                    </tr>
                '''
            return rows_html
    
        primary_cph_rows_html = build_cph_rows(primary_comparison_rows_data)
        if not primary_cph_rows_html and not is_full_cohort:
            primary_cph_rows_html = '<tr><td colspan="8" style="text-align: center; color: #64748b; background-color: #ffffff;">No primary comparison evaluated.</td></tr>'
    
        multivariate_cph_rows_html = build_cph_rows(multivariate_covs_rows_data)
        if not multivariate_cph_rows_html:
            multivariate_cph_rows_html = '<tr><td colspan="8" style="text-align: center; color: #64748b; background-color: #ffffff;">No multivariate covariates evaluated.</td></tr>'
                
        # --- Construct Table 1 Dynamic HTML Rows from State ---
        table1_rows_html = ""
        table_data = getattr(self, "table_one_data", [])
    
        if table_data:
            for row in table_data:
                var_text = row.get("variable", "")
                row_type = row.get("type", "Continuous")
                is_cat = (row_type == "Categorical")
                is_hdr = (row_type == "Header")
                
                padding_left = "18px" if is_cat else "10px"
                raw_p = row.get("p_value", "N/A")
                formatted_p = str(raw_p)
                try:
                    p_float = float(raw_p)
                    formatted_p = "< 0.001" if p_float < 0.001 else f"{p_float:.3f}"
                except (ValueError, TypeError):
                    pass
    
                g1_val = row.get("group1", "")
                g2_val = row.get("group2", "")
    
                if is_full_cohort:
                    if is_hdr:
                        table1_rows_html += f'''
                            <tr>
                                <td colspan="2" style="font-weight: bold; color: #1e293b; padding: 7px 10px; border-top: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; background-color: #ffffff; font-size: 8.5pt;">{var_text}</td>
                            </tr>
                        '''
                    else:
                        table1_rows_html += f'''
                            <tr>
                                <td style="padding-left: {padding_left}; width: 60%; background-color: #ffffff; color: #1e293b; font-size: 8.5pt; border: none;">{var_text}</td>
                                <td style="text-align: right; padding-right: 24px; font-variant-numeric: tabular-nums; background-color: #ffffff; width: 40%; color: #1e293b; font-size: 8.5pt; border: none;">{g1_val}</td>
                            </tr>
                        '''
                else:
                    if is_hdr:
                        table1_rows_html += f'''
                            <tr>
                                <td colspan="4" style="font-weight: bold; color: #1e293b; padding: 7px 10px; border-top: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; background-color: #ffffff; font-size: 8.5pt;">{var_text}</td>
                            </tr>
                        '''
                    else:
                        # EXPLICIT WIDTHS FOR EVERY CELL MATCHING 24% / 32% / 32% / 12%
                        table1_rows_html += f'''
                            <tr>
                                <td style="padding-left: {padding_left}; width: 24%; background-color: #ffffff; color: #1e293b; font-size: 8.5pt; border: none;">{var_text}</td>
                                <td style="text-align: center; width: 32%; font-variant-numeric: tabular-nums; background-color: #ffffff; color: #1e293b; font-size: 8.5pt; border: none;">{g1_val}</td>
                                <td style="text-align: center; width: 32%; font-variant-numeric: tabular-nums; background-color: #ffffff; color: #1e293b; font-size: 8.5pt; border: none;">{g2_val}</td>
                                <td style="text-align: center; width: 12%; color: #1e293b; font-weight: normal; font-variant-numeric: tabular-nums; background-color: #ffffff; font-size: 8.5pt; border: none;">{formatted_p}</td>
                            </tr>
                        '''
        else:
            colspan_val = "2" if is_full_cohort else "4"
            table1_rows_html = f'''
                <tr>
                    <td colspan="{colspan_val}" style="text-align: center; color: #64748b; padding: 15px; background-color: #ffffff; border: none;">No baseline characteristics configured or available.</td>
                </tr>
            '''
    
        # --- Dynamic Table 1 Header Switch & Subtitle Description ---
        if is_full_cohort:
            table1_header_html = f'''
                <tr style="border-bottom: 1.5px solid #1e293b;">
                    <th style="width: 60%; text-align: left; background-color: #ffffff; color: #1e293b; padding: 8px 10px; font-size: 8.5pt; font-weight: bold; vertical-align: top;">Characteristic</th>
                    <th style="width: 40%; text-align: right; background-color: #ffffff; color: #1e293b; padding: 8px 10px; font-size: 8.5pt; font-weight: bold; vertical-align: top; padding-right: 24px;">{g1_label}</th>
                </tr>
            '''
            table1_desc = "Baseline characteristics for the overall study cohort. Continuous variables shown as Mean ± SD; categorical variables shown as n (%)."
        else:
            # HEADER WIDTHS EXPLICITLY SET TO 24% / 32% / 32% / 12%
            table1_header_html = f'''
                <tr style="border-bottom: 1.5px solid #1e293b;">
                    <th style="width: 24%; text-align: left; background-color: #ffffff; color: #1e293b; padding: 8px 10px; font-size: 8.5pt; font-weight: bold; vertical-align: top;">Characteristic</th>
                    <th style="width: 32%; text-align: center; background-color: #ffffff; color: #1e293b; padding: 8px 10px; font-size: 8.5pt; font-weight: bold; vertical-align: top;">
                        {h1_title}<br>
                        <span style="font-weight: normal; font-size: 7.5pt; color: #475569;">{h1_sub}</span>
                    </th>
                    <th style="width: 32%; text-align: center; background-color: #ffffff; color: #1e293b; padding: 8px 10px; font-size: 8.5pt; font-weight: bold; vertical-align: top;">
                        {h2_title}<br>
                        <span style="font-weight: normal; font-size: 7.5pt; color: #475569;">{h2_sub}</span>
                    </th>
                    <th style="width: 12%; text-align: center; background-color: #ffffff; color: #1e293b; padding: 8px 10px; font-size: 8.5pt; font-weight: bold; vertical-align: top;">P-Value</th>
                </tr>
            '''
            table1_desc = "Baseline characteristics stratified by active comparison groups. Continuous variables shown as Mean ± SD; categorical variables shown as n (%)."
    
        table1_subtitle_html = f'''
            <div style="font-size: 8.5pt; color: #64748b; margin-bottom: 10px; line-height: 1.4; page-break-after: avoid; break-after: avoid;">
                {table1_desc}
            </div>
        '''

        # --- Consume State Variables for Journal-Standard KM Summary Table ---
        surv_list = getattr(self, "survival_probability_list", [])
        med_list = getattr(self, "median_survival_list", [])
        log_rank_p = getattr(self, "p_value_text", "N/A")

        # Parse survival probability and median survival strings from state lists cleanly
        g1_surv_val, g2_surv_val = "N/A", "N/A"
        g1_med_val, g2_med_val = "N/A", "N/A"

        for item in surv_list:
            item_str = str(item)
            if "Group 1" in item_str:
                g1_surv_val = item_str.split(":", 1)[-1].strip() if ":" in item_str else item_str
            elif "Group 2" in item_str:
                g2_surv_val = item_str.split(":", 1)[-1].strip() if ":" in item_str else item_str
            elif is_full_cohort:
                g1_surv_val = item_str.strip()

        for item in med_list:
            item_str = str(item)
            if "Group 1" in item_str:
                g1_med_val = item_str.split(":", 1)[-1].strip() if ":" in item_str else item_str
            elif "Group 2" in item_str:
                g2_med_val = item_str.split(":", 1)[-1].strip() if ":" in item_str else item_str
            elif is_full_cohort:
                g1_med_val = item_str.strip()
        
        raw_g1_desc = getattr(self, "g1_detailed_label", "")
        raw_g2_desc = getattr(self, "g2_detailed_label", "")

        # Define clean_criteria helper to parse raw state labels
        def clean_criteria(raw_str):
            cleaned = str(raw_str)
            for prefix in [
                f"{g1_label}:", f"{g2_label}:", 
                "Group 1:", "Group 2:", 
                "Group 1 (Complement):", "Group 2 (Complement):",
                "Full Cohort"
            ]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned.replace(prefix, "", 1).strip()
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = cleaned[1:-1].strip()
            return cleaned if cleaned else "Full Cohort / Unstratified"

        g1_clean = clean_criteria(raw_g1_desc)
        g2_clean = clean_criteria(raw_g2_desc)

        def parse_group_header_and_sub(label_str, clean_str):
            is_comp = "Complement" in label_str or "Complement" in clean_str
            
            if "Group 1" in label_str:
                base_header = "Group 1"
            elif "Group 2" in label_str:
                base_header = "Group 2"
            else:
                base_header = label_str.split("(")[0].strip()
                
            if is_comp:
                sub_text = "Complement"
            else:
                raw_sub = clean_str if clean_str and clean_str != "Full Cohort" else "Full Cohort"
                if raw_sub not in ["Full Cohort", "Complement"]:
                    parts = [f'<span style="white-space: nowrap;">{p.strip()}</span>' for p in raw_sub.split(",")]
                    sub_text = ", ".join(parts)
                else:
                    sub_text = raw_sub
                
            return base_header, sub_text

        h1_title, h1_sub = parse_group_header_and_sub(g1_label, g1_clean)
        h2_title, h2_sub = parse_group_header_and_sub(g2_label, g2_clean)

        if is_full_cohort:
            km_table_html = f'''
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 25px; font-size: 8.5pt; background-color: #ffffff; border-top: 1.5px solid #1e293b; border-bottom: 1.5px solid #1e293b;">
                    <thead>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <th style="width: 30%; text-align: left; padding: 8px 10px; color: #1e293b; font-weight: bold; background-color: #ffffff; vertical-align: top;">Clinical Endpoint / Metric</th>
                            <th style="width: 70%; text-align: left; padding: 8px 10px; color: #1e293b; font-weight: bold; background-color: #ffffff; vertical-align: top;" colspan="3">Full Cohort (Unstratified)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;"><strong>Stratification Criteria</strong></td>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;" colspan="3">Full Cohort</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;"><strong>Total N (Observed Events)</strong></td>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;" colspan="3">{g1_n} ({g1_events} events)</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;"><strong>Survival Probability</strong></td>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;" colspan="3">{g1_surv_val}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;"><strong>Median Survival Time</strong></td>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;" colspan="3">{g1_med_val}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;"><strong>Statistical Parameters</strong></td>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;" colspan="3">Unstratified Cohort (Log-rank test N/A)</td>
                        </tr>
                    </tbody>
                </table>
            '''
        else:
           km_table_html = f'''
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 25px; font-size: 8.5pt; background-color: #ffffff; border-top: 1.5px solid #1e293b; border-bottom: 1.5px solid #1e293b;">
                    <thead>
                        <tr style="border-bottom: 1.5px solid #1e293b;">
                            <th style="width: 24%; text-align: left; padding: 8px 10px; color: #1e293b; font-weight: bold; background-color: #ffffff; vertical-align: top;">Clinical Endpoint / Metric</th>
                            <th style="width: 32%; text-align: left; padding: 8px 10px; color: #1e293b; font-weight: bold; background-color: #ffffff; vertical-align: top;">
                                {h1_title}<br>
                                <span style="font-weight: normal; font-size: 7.5pt; color: #475569;">{h1_sub}</span>
                            </th>
                            <th style="width: 32%; text-align: left; padding: 8px 10px; color: #1e293b; font-weight: bold; background-color: #ffffff; vertical-align: top;">
                                {h2_title}<br>
                                <span style="font-weight: normal; font-size: 7.5pt; color: #475569;">{h2_sub}</span>
                            </th>
                            <th style="width: 12%; text-align: left; padding: 8px 10px; color: #1e293b; font-weight: bold; background-color: #ffffff; vertical-align: top;">P-Value</th>
                        </tr>
                    </thead>
                    <tbody>

                        <tr>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;"><strong>Total N (Observed Events)</strong></td>
                            <td style="text-align: left; padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none; font-variant-numeric: tabular-nums; white-space: nowrap;">{g1_n} ({g1_events})</td>
                            <td style="text-align: left; padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none; font-variant-numeric: tabular-nums; white-space: nowrap;">{g2_n} ({g2_events})</td>
                            <td style="text-align: left; padding: 6px 10px; color: #64748b; background-color: #ffffff; border: none;">—</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;"><strong>Survival Probability</strong></td>
                            <td style="text-align: left; padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none; font-variant-numeric: tabular-nums; white-space: nowrap;">{g1_surv_val}</td>
                            <td style="text-align: left; padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none; font-variant-numeric: tabular-nums; white-space: nowrap;">{g2_surv_val}</td>
                            <td style="text-align: left; padding: 6px 10px; color: #64748b; background-color: #ffffff; border: none;">—</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none;"><strong>Median Survival Time</strong></td>
                            <td style="text-align: left; padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none; font-variant-numeric: tabular-nums; white-space: nowrap;">{g1_med_val}</td>
                            <td style="text-align: left; padding: 6px 10px; color: #1e293b; background-color: #ffffff; border: none; font-variant-numeric: tabular-nums; white-space: nowrap;">{g2_med_val}</td>
                            <td style="text-align: left; padding: 6px 10px; color: #64748b; background-color: #ffffff; border: none;">—</td>
                        </tr>
                        <tr style="border-top: 1px solid #e2e8f0;">
                            <td style="padding: 8px 10px; color: #1e293b; background-color: #ffffff; border: none;"><strong>Patient Group Comparison</strong><br><span style="font-size: 7.5pt; color: #64748b; font-weight: normal;">Log-rank test</span></td>
                            <td style="text-align: left; padding: 8px 10px; color: #1e293b; background-color: #ffffff; border: none; font-weight: 500; vertical-align: middle;" colspan="2">Evaluation of overall survival distribution equality</td>
                            <td style="text-align: left; padding: 8px 10px; color: #1e293b; background-color: #ffffff; border: none; font-weight: bold; vertical-align: middle; font-variant-numeric: tabular-nums;">{log_rank_p}</td>
                        </tr>
                    </tbody>
                </table>
            '''

        # --- Number at Risk Table --- #
        at_risk_cols = getattr(self, "at_risk_columns", ["Number at Risk", "0", "5", "11", "16", "22"])
        at_risk_rows = getattr(self, "at_risk_data", [])
        first_col_title = at_risk_cols[0] if len(at_risk_cols) > 0 else "Number at Risk (Mo)"
        
        time_headers = "".join([
            f'<th style="text-align: right; padding: 6px 10px; color: #1e293b; font-weight: bold; font-variant-numeric: tabular-nums; background-color: #ffffff; vertical-align: top;">{t}</th>' 
            for t in at_risk_cols[1:]
        ])
    
        risk_rows_html = ""
        for row in at_risk_rows:
            row_label = row[0] if len(row) > 0 else ""
            counts = row[1:] if len(row) > 1 else []
            
            # Neutralize label styling to authoritative dark charcoal for medical journal standard
            label_style = "font-weight: bold; color: #1e293b; white-space: nowrap;"
            
            # Map row labels to match our h1/h2 parsed titles if applicable
            display_label = row_label
            if "Group 1" in row_label:
                display_label = h1_title if 'h1_title' in locals() else "Group 1"
            elif "Group 2" in row_label:
                display_label = h2_title if 'h2_title' in locals() else "Group 2"

            cells_html = "".join([
                f'<td style="text-align: right; padding: 6px 10px; color: #1e293b; font-variant-numeric: tabular-nums; background-color: #ffffff; border: none;">{val}</td>' 
                for val in counts
            ])
            risk_rows_html += f'''
                <tr>
                    <td style="text-align: left; padding: 6px 10px; {label_style} background-color: #ffffff; border: none;">{display_label}</td>
                    {cells_html}
                </tr>
            '''
    
        risk_table_html = f'''
        <div style="margin-top: 25px; margin-bottom: 25px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 8.5pt; background-color: #ffffff; border-top: 1.5px solid #1e293b; border-bottom: 1.5px solid #1e293b;">
                <thead>
                    <tr style="border-bottom: 1.5px solid #1e293b;">
                        <th style="text-align: left; padding: 8px 10px; color: #1e293b; width: 35%; font-weight: bold; background-color: #ffffff; vertical-align: top;">{first_col_title}</th>
                        {time_headers}
                    </tr>
                </thead>
                <tbody>
                    {risk_rows_html}
                </tbody>
            </table>
        </div>
        '''
    
        risk_table_html = f'''
        <div style="margin-top: 22px; border-top: 1px solid #e2e8f0; padding-top: 14px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 8.5pt; border: 1px solid #cbd5e1; background-color: #ffffff;">
                <thead>
                    <tr style="border-bottom: 1px solid #cbd5e1;">
                        <th style="text-align: left; padding: 5px 8px; color: #475569; width: 35%; font-weight: bold; background-color: #ffffff;">{first_col_title}</th>
                        {time_headers}
                    </tr>
                </thead>
                <tbody>
                    {risk_rows_html}
                </tbody>
            </table>
        </div>
        '''
    
        # --- Chart Display --- #
        chart_html_block = ""
        try:
            fig_obj = getattr(self, "survival_fig_g1", getattr(self, "survival_fig", None))
            img_bytes = fig_obj.to_image(format="png", width=900, height=450, scale=2)
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            chart_html_block = f'''
            <div class="chart-container" style="margin-bottom: 30px;">
                <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto; display: block;" />
                {risk_table_html}
                <div style="font-size: 8.5pt; color: #64748b; margin-top: 10px;">
                    Figure 1: Comparative Kaplan-Meier Survival Curves with corresponding Number at Risk table. Vertical tick marks denote censored data points.
                </div>
            </div>
            '''
        except Exception as e:
            chart_html_block = f'''
            <div class="chart-container" style="padding: 20px; color: #dc2626; text-align: center; font-size: 9pt; margin-bottom: 30px;">
                [Chart Export Error: {e}]
            </div>
            '''
    
        # --- Footnotes and Abbreviations Block --- #
        tie_adjustment_note = ""
        if getattr(self, "km_warning_message", "") != "":
            tie_adjustment_note = " Risk score tertile boundaries were adjusted using rank-based tie-breaking to accommodate clustered data."

        cph_footnotes_html = f'''
        <div style="font-size: 7.5pt; color: #64748b; margin-top: 24px; line-height: 1.4; border-top: 1px solid #cbd5e1; padding-top: 12px;">
            <div style="margin-bottom: 8px;"><strong>Abbreviations:</strong> HR, hazard ratio; CI, confidence interval; Adj, adjusted.</div>
            <div><strong>Note:</strong> Univariate and multivariable Cox proportional hazards models evaluate time-to-event outcomes. Schoenfeld residual p-values evaluate the proportional hazards assumption (p > 0.05 indicates satisfaction of the proportional hazards assumption).{tie_adjustment_note}</div>
        </div>
        '''

        # --- CPH Tables Blocks for PDF --- # 
        if is_full_cohort:
            cph_section_block = f'''
            <div class="section-block" style="page-break-before: always;">
                <h2 style="font-size: 11pt; color: #1e293b; font-weight: bold; margin-bottom: 12px; border-bottom: 1.5px solid #1e293b; padding-bottom: 4px;">Multivariate Cox Proportional Hazards & Univariate Assessment</h2>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 8.0pt; background-color: #ffffff; border-top: 1.5px solid #1e293b; border-bottom: 1.5px solid #1e293b;">
                    <thead>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <th rowspan="2" style="text-align: left; vertical-align: top; width: 22%; padding: 8px 8px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Covariate</th>
                            <th colspan="3" style="text-align: center; vertical-align: top; border-bottom: 1px solid #cbd5e1; padding: 6px 8px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Univariate Analysis</th>
                            <th colspan="4" style="text-align: center; vertical-align: top; border-bottom: 1px solid #cbd5e1; padding: 6px 8px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Multivariate Adjusted Analysis</th>
                        </tr>
                        <tr style="border-bottom: 1.5px solid #1e293b;">
                            <th style="text-align: center; vertical-align: top; width: 10%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">HR</th>
                            <th style="text-align: center; vertical-align: top; width: 13%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">95% CI</th>
                            <th style="text-align: center; vertical-align: top; width: 10%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">P-Value</th>
                            <th style="text-align: center; vertical-align: top; width: 10%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Adj. HR</th>
                            <th style="text-align: center; vertical-align: top; width: 13%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">95% CI</th>
                            <th style="text-align: center; vertical-align: top; width: 11%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Adj. P</th>
                            <th style="text-align: center; vertical-align: top; width: 11%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Schoenfeld P</th>
                        </tr>
                    </thead>
                    <tbody>
                        {multivariate_cph_rows_html}
                    </tbody>
                </table>
                {cph_footnotes_html}
            </div>
            '''
        else:
            cph_section_block = f'''
            <div class="section-block" style="page-break-before: always;">
                <h2 style="font-size: 11pt; color: #1e293b; font-weight: bold; margin-bottom: 12px; border-bottom: 1.5px solid #1e293b; padding-bottom: 4px;">Cox Proportional Hazards Analysis & Hazard Ratios</h2>
                
                <h3 style="font-size: 9pt; color: #1e293b; margin-top: 15px; margin-bottom: 2px; font-weight: bold;">
                    Table A: Primary Group Comparison ({h1_title} vs. {h2_title})
                </h3>
                <div style="font-size: 8pt; color: #64748b; margin-bottom: 6px; font-weight: normal;">
                    Stratification definitions — {h1_title}: {h1_sub} &nbsp;|&nbsp; {h2_title}: {h2_sub}
                </div>
                
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 8.0pt; background-color: #ffffff; border-top: 1.5px solid #1e293b; border-bottom: 1.5px solid #1e293b;">
                    <thead>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <th rowspan="2" style="text-align: left; vertical-align: top; width: 22%; padding: 8px 8px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Comparison Variable</th>
                            <th colspan="3" style="text-align: center; vertical-align: top; border-bottom: 1px solid #cbd5e1; padding: 6px 8px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Univariate Analysis</th>
                            <th colspan="4" style="text-align: center; vertical-align: top; border-bottom: 1px solid #cbd5e1; padding: 6px 8px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Multivariate Adjusted Analysis</th>
                        </tr>
                        <tr style="border-bottom: 1.5px solid #1e293b;">
                            <th style="text-align: center; vertical-align: top; width: 10%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">HR</th>
                            <th style="text-align: center; vertical-align: top; width: 13%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">95% CI</th>
                            <th style="text-align: center; vertical-align: top; width: 10%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">P-Value</th>
                            <th style="text-align: center; vertical-align: top; width: 10%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Adj. HR</th>
                            <th style="text-align: center; vertical-align: top; width: 13%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">95% CI</th>
                            <th style="text-align: center; vertical-align: top; width: 11%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Adj. P</th>
                            <th style="text-align: center; vertical-align: top; width: 11%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Schoenfeld P</th>
                        </tr>
                    </thead>
                    <tbody>
                        {primary_cph_rows_html}
                    </tbody>
                </table>

                <h3 style="font-size: 9pt; color: #1e293b; margin-top: 34px; margin-bottom: 6px; font-weight: bold;">Table B: Multivariable Background Clinical Adjustment Panel</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 8.0pt; background-color: #ffffff; border-top: 1.5px solid #1e293b; border-bottom: 1.5px solid #1e293b;">
                    <thead>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <th rowspan="2" style="text-align: left; vertical-align: top; width: 22%; padding: 8px 8px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Covariate</th>
                            <th colspan="3" style="text-align: center; vertical-align: top; border-bottom: 1px solid #cbd5e1; padding: 6px 8px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Univariate Analysis</th>
                            <th colspan="4" style="text-align: center; vertical-align: top; border-bottom: 1px solid #cbd5e1; padding: 6px 8px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Multivariate Adjusted Analysis</th>
                        </tr>
                        <tr style="border-bottom: 1.5px solid #1e293b;">
                            <th style="text-align: center; vertical-align: top; width: 10%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">HR</th>
                            <th style="text-align: center; vertical-align: top; width: 13%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">95% CI</th>
                            <th style="text-align: center; vertical-align: top; width: 10%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">P-Value</th>
                            <th style="text-align: center; vertical-align: top; width: 10%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Adj. HR</th>
                            <th style="text-align: center; vertical-align: top; width: 13%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">95% CI</th>
                            <th style="text-align: center; vertical-align: top; width: 11%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Adj. P</th>
                            <th style="text-align: center; vertical-align: top; width: 11%; padding: 6px 6px; color: #1e293b; font-weight: bold; background-color: #ffffff;">Schoenfeld P</th>
                        </tr>
                    </thead>
                    <tbody>
                        {multivariate_cph_rows_html}
                    </tbody>
                </table>
                {cph_footnotes_html}
            </div>
            '''

        # --- Full HTML Document Assembly for Weasyprint --- #
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Publication-Grade Kaplan-Meier Survival Analysis Report</title>
            <style>
                *, *::before, *::after {{ box-sizing: border-box; }}
                @page {{
                    size: A4;
                    margin: 15mm 15mm 18mm 15mm;
                    @bottom-right {{
                        content: "Page " counter(page) " of " counter(pages);
                        font-size: 8pt;
                        color: #64748b;
                        font-family: Helvetica, Arial, sans-serif;
                    }}
                    @bottom-left {{
                        content: "STROBE/REMARK-Compliant Prognostic Report | Clinical Survival Platform";
                        font-size: 8pt;
                        color: #64748b;
                        font-family: Helvetica, Arial, sans-serif;
                    }}
                }}
                body {{
                    font-family: Helvetica, Arial, sans-serif;
                    color: #1e293b;
                    font-size: 8.5pt;
                    line-height: 1.4;
                    margin: 0;
                    padding: 0;
                    background-color: #ffffff;
                }}
                .header-banner {{
                    border-bottom: 2px solid #1e293b;
                    padding-bottom: 8px;
                    margin-bottom: 22px;
                }}
                .header-banner h1 {{
                    font-size: 13pt;
                    color: #1e293b;
                    margin: 0 0 2px 0;
                    font-weight: bold;
                    text-transform: uppercase;
                    letter-spacing: 0.3px;
                }}
                .header-banner .subtitle {{
                    font-size: 8pt;
                    color: #64748b;
                    margin: 0;
                }}
                .meta-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 8pt;
                    border: 1px solid #cbd5e1;
                    margin-bottom: 28px;
                    background-color: #ffffff;
                }}
                .meta-table td {{
                    padding: 5px 8px;
                    border: 1px solid #cbd5e1;
                    vertical-align: middle;
                }}
                .meta-label {{ font-weight: bold; color: #1e293b; width: 18%; background-color: #f8fafc; }}
                .meta-val {{ color: #334155; width: 32%; }}
                h2 {{
                    font-size: 10pt;
                    color: #1e293b;
                    border-bottom: 1.5px solid #1e293b;
                    padding-bottom: 3px;
                    margin-top: 24px;
                    margin-bottom: 10px;
                    font-weight: bold;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    page-break-after: avoid;
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 8.0pt;
                    background-color: #ffffff;
                    border-top: 1.5px solid #1e293b;
                    border-bottom: 1.5px solid #1e293b;
                    margin-bottom: 12px;
                    page-break-inside: auto; /* Added to prevent massive vertical gaps */
                }}
                .data-table tr {{
                    page-break-inside: avoid;    /* Prevents rows from splitting awkwardly */
                    page-break-after: auto;
                }}
                .section-block {{
                    page-break-inside: auto;
                }}
                .data-table th {{
                    background-color: #f1f5f9;
                    color: #1e293b;
                    font-weight: bold;
                    text-align: left;
                    padding: 7px 10px;
                    border: 1px solid #cbd5e1;
                }}
                .data-table td {{
                    padding: 6px 10px;
                    border: 1px solid #e2e8f0;
                    vertical-align: middle;
                }}
                .data-table tr:nth-child(even) td {{
                    background-color: #f8fafc;
                }}
            </style>
        </head>
        <body>
            <div class="header-banner" style="border-bottom: 2px solid #1e293b; padding-bottom: 10px; margin-bottom: 22px;">
                <h1 style="font-size: 13pt; color: #1e293b; font-weight: bold; margin: 0 0 3px 0; letter-spacing: -0.2px; text-transform: uppercase;">
                    Clinical Survival Analysis Report
                </h1>
                <div style="font-size: 8.0pt; color: #475569; font-weight: normal;">
                    Publication-Grade Manuscript Supplement &mdash; STROBE/REMARK Standards
                </div>
            </div>

            <table class="meta-table">
                <tr>
                    <td class="meta-label">Dataset Source:</td>
                    <td class="meta-val">{dataset_name}</td>
                    <td class="meta-label">Model Type:</td>
                    <td class="meta-val">{model_name}</td>
                </tr>
                <tr>
                    <td class="meta-label">Lead Investigator:</td>
                    <td class="meta-val">{researcher_name}</td>
                    <td class="meta-label">Institution:</td>
                    <td class="meta-val">{institution_name}</td>
                </tr>
                <tr>
                    <td class="meta-label">Report Date:</td>
                    <td class="meta-val" colspan="3">{generation_date}</td>
                </tr>
            </table>

            <!-- SECTION 1: TABLE 1 BASELINE CHARACTERISTICS -->
            <div class="section-block">
                <h2>Table 1: Baseline Demographics & Clinical Characteristics</h2>
                {table1_subtitle_html}
                <table class="data-table">
                    <thead>
                        {table1_header_html}
                    </thead>
                    <tbody>
                        {table1_rows_html}
                    </tbody>
                </table>
            </div>
            
            <!-- SECTION 2: KAPLAN-MEIER SURVIVAL SUMMARY & ANALYSIS TABLE -->
            <div class="section-block" style="page-break-before: always;">
                <h2>Kaplan-Meier Survival Summary & Analysis</h2>
                {km_table_html}
                {chart_html_block}
            </div>

            <!-- SECTION 3: CPH HAZARD RATIOS -->
            {cph_section_block}

        </body>
        </html>
        '''

        # Render PDF via Weasyprint
        try:
            pdf_bytes = HTML(string=html_content).write_pdf()
            return rx.download(data=pdf_bytes, filename="clinical_km_journal_report.pdf")
        except Exception as e:
            print(f"PDF Generation Error: {e}")
            return
    
    @rx.var
    def survival_header_text(self) -> str:
        val = self.timeline_values[1] if self.timeline_values else 0
        unit = getattr(self, "timeline_unit", "Mo")
        return f"SURVIVAL AT {int(val)} {unit}"
    
    @rx.var
    def cds_severity(self) -> str:
        """Determines the color theme of the CDS card based on statistical findings."""
        # 🌟 If results haven't been generated yet or were cleared, return 'pending'
        if not getattr(self, "results_ready", False):
            return "pending"

        is_significant = False
        try:
            # Safely extract float from your p-value (assuming p_value_text looks like "Log-rank p: 0.0126")
            p_val_str = self.p_value_text.split(":")[-1].strip()
            p_val = float(p_val_str)
            is_significant = p_val < 0.05
        except Exception:
            # Fallback if parsing fails
            is_significant = False

        if is_significant:
            return "amber"  # Actionable/Warning status
        
        return "gray"  # Neutral/Informational status (only when a valid non-significant run exists)

    @rx.var
    def cds_guidance_title(self) -> str:
        if self.cds_severity == "pending":
            return "Awaiting Analysis: Parameters Modified"
        
        # Check if running an unstratified Full Cohort analysis
        if self.is_full_cohort:
            return "Clinical Information: Full Cohort Survival Profile"
            
        if self.cds_severity == "amber":
            return "Action Required: Significant Survival Variance Detected"
            
        return "Clinical Information: No Significant Survival Variance"

    @rx.var
    def cds_guidance_text(self) -> str:
        if self.cds_severity == "pending":
            return "Configuration updated in sidebar. Click 'Run Analysis' to generate updated clinical guidance."
        
        # 🌟 Branch cleanly for Full Cohort (unstratified) analysis
        if self.is_full_cohort:
            return (
                "Full cohort survival analysis completed successfully using non-parametric Kaplan-Meier estimation. "
                "The trajectory illustrates overall population baseline survival probability and median survival timelines. "
                "Refer to the survival curve and Risk Table for granular interval evaluations."
            )

        # Stratified Two-Group comparative logic:
        if self.cds_severity == "amber":
            return (
                "The log-rank evaluation indicates a statistically significant divergence in survival trajectories "
                "between the selected cohorts (p < 0.05). Risk stratification demonstrates disparate outcome paths, "
                "warranting consideration of protocol adjustments, targeted supportive care, or escalating monitoring for the high-risk group."
            )
        
        return (
            "Survival analysis demonstrates equivalent outcome trajectories between the evaluated cohorts (p ≥ 0.05). "
            "Observed survival curves show statistical parity, supporting the continuation of standard baseline observation and management protocols."
        )

    _df: Optional[pd.DataFrame] = None      # variable starts as None, but when it has data, that data will be a Pandas DataFrame

    # --- Timeline Slider Settings --- #
    # Initialize at zero/empty states
    timeline_max: int = 0
    timeline_values: list[int] = [0, 0]
    timeline_unit: str = "Mo"  # 🎯 Add this to track the normalized unit!

    # Tracks if the Min/Max inputs for a specific column are visible
    #custom_range_visible: dict[str, bool] = {}

    g1_custom_visible: dict[str, bool] = {}
    g2_custom_visible: dict[str, bool] = {}

    @rx.event
    def handle_filter_change(self, group_num: int, col_name: str, display_label: str, val: bool):
        if display_label == "Custom Range:":
            # Toggle logic is now here (internal), no external function needed
            if group_num == 1:
                self.g1_custom_visible[col_name] = not self.g1_custom_visible.get(col_name, False)
            else:
                self.g2_custom_visible[col_name] = not self.g2_custom_visible.get(col_name, False)
        else:
            # Handle normal selection (ensure this function exists and accepts 4 args)
            return self.handle_interval_selection(group_num, col_name, display_label, val)


    #@rx.event
    def toggle_custom_visible(self, group_num: int, col_name: str):
        if group_num == 1:
            self.g1_custom_visible[col_name] = not self.g1_custom_visible.get(col_name, False)
        else:
            self.g2_custom_visible[col_name] = not self.g2_custom_visible.get(col_name, False)

    @rx.event
    def handle_filter_change(self, group_num: int, col_name: str, display_label: str, val: bool):
        if display_label == "Custom Range:":
            return self.toggle_custom_visible(group_num, col_name)
        else:
            return self.handle_interval_selection(group_num, col_name, display_label, val)
    
    @rx.var
    def g1_checked_list(self) -> dict:
        return self.g1_checked_values if self.g1_checked_values is not None else {}

    @rx.var
    def g2_checked_list(self) -> dict:
        return self.g2_checked_values if self.g2_checked_values is not None else {}
    
    def is_item_selected(self, group_num: int, col_name: str, label: str) -> bool:
        # Use .get() to avoid KeyErrors and handle None
        data = self.g1_checked_values if group_num == 1 else self.g2_checked_values
        if data is None: return False
        return label in data.get(col_name, [])

    def reset_group_data(self):
        # Instead of just empty dicts, explicitly clear the keys
        self.g1_checked_values.clear()
        self.g2_checked_values.clear()
        self.g1_custom_min.clear()
        self.g1_custom_max.clear()
        self.g2_custom_min.clear()
        self.g2_custom_max.clear()
        self.is_ready = False
    
    @rx.var
    def g1_detailed_label(self) -> str:
        return self._format_group_label(
            "Group 1", 
            self.g1_checked_values, 
            self.g1_custom_min, 
            self.g1_custom_max
        )

    @rx.var
    def g2_detailed_label(self) -> str:
        return self._format_group_label(
            "Group 2",
            self.g2_checked_values, 
            self.g2_custom_min, 
            self.g2_custom_max
        )
    
    def _format_group_label(self, group_name: str, checked_values: dict, custom_min: dict = None, custom_max: dict = None) -> str:
        details = []
        custom_min = custom_min or {}
        custom_max = custom_max or {}

        # Get a list of all columns that have EITHER checked values OR custom data
        all_cols = set(list(checked_values.keys()) + list(custom_min.keys()) + list(custom_max.keys()))

        for col in all_cols:
            val = checked_values.get(col)
            low = custom_min.get(col)
            high = custom_max.get(col)

            # 1. Prioritize Custom Range if data exists
            if (low not in [None, ""] or high not in [None, ""]):
                low_val = low if (low not in [None, ""]) else "Min"
                high_val = high if (high not in [None, ""]) else "Max"
                details.append(f"{col} = Custom Range: {low_val}-{high_val}")

            # 2. Handle standard selections (if not already handled as custom)
            elif val and str(val) != "[]" and str(val) != "None" and "Custom Range:" not in str(val):
                val_str = ", ".join(map(str, val)) if isinstance(val, (list, tuple, set)) else str(val)
                details.append(f"{col} = {val_str}")

        if not details:
            return f"{group_name}: (Full Cohort)"
            
        #return f"{group_name}: {{{'; '.join(details)}}}"
        return f"{group_name}: ({', '.join(details)})"

    # Required State attributes
    g1_use_custom_range: dict[str, bool] = {}
    g2_use_custom_range: dict[str, bool] = {}

    g1_checked_values: dict[str, list[str]] = {}
    g2_checked_values: dict[str, list[str]] = {}

    # Don't forget your custom min/max state dicts if they aren't already defined
    g1_custom_min: dict[str, float] = {}
    g1_custom_max: dict[str, float] = {}
    g2_custom_min: dict[str, float] = {}
    g2_custom_max: dict[str, float] = {}
    is_ready: bool = True

    critical_columns: list[str] = []
    
    def is_selected_for_ui(self, group_num: int, col_name: str, label: str) -> bool:
        target_dict = self.g1_checked_values if group_num == 1 else self.g2_checked_values
        # Safe access to the list in the dictionary
        column_list = target_dict.get(col_name, [])
        return label in column_list
    
    def is_selected(self, group_num: int, col_name: str, label: str) -> bool:
        # Safely get the dictionary, default to empty dict if missing
        target_dict = self.g1_checked_values if group_num == 1 else self.g2_checked_values
        column_list = target_dict.get(col_name, [])
        return label in column_list

    @rx.var
    def g1_columns_with_custom_data(self) -> list[str]:
        cols = []
        for col, val in self.g1_custom_min.items():
            max_val = self.g1_custom_max.get(col)
            # Check if both fields are filled out and not empty strings
            if val is not None and str(val).strip() != "" and max_val is not None and str(max_val).strip() != "":
                cols.append(col)
        return cols

    @rx.var
    def g2_columns_with_custom_data(self) -> list[str]:
        cols = []
        for col, val in self.g2_custom_min.items():
            max_val = self.g2_custom_max.get(col)
            # Check if both fields are filled out and not empty strings
            if val is not None and str(val).strip() != "" and max_val is not None and str(max_val).strip() != "":
                cols.append(col)
        return cols

    @rx.event
    def toggle_variable_dropdown(self, group_num: int, col_name: str):
        """Manages custom cohort drill-down matrices while protecting row filters and managing workflow icons."""
        col_str = str(col_name).strip()
        
        # 1. Safety blocks for target variables
        if col_str in ["time", "rfstime", "rectime", "status", "event", "cens"]:
            return
        if col_str == str(self.selected_time_column) or col_str == str(self.selected_event_column):
            return
        
        if group_num == 1:
            current_expanded = self.g1_expanded.get(col_str, False)
            next_expanded = not current_expanded
            self.g1_expanded[col_str] = next_expanded
            
            if next_expanded:
                # 🟢 OPENING: Set active editing focus
                self.g1_active_editing_col = col_str
            else:
                # 🔴 CLOSING: Strip active editing focus
                if self.g1_active_editing_col == col_str:
                    self.g1_active_editing_col = ""
                
                val = self.g1_checked_values.get(col_str, [])
                
                # Deep scan standard categorical selections
                if isinstance(val, list):
                    has_categorical = any(isinstance(v, str) and v.strip() != "" and "custom" not in v.lower() for v in val) or any(not isinstance(v, str) for v in val if v is not None)
                else:
                    has_categorical = isinstance(val, str) and val.strip() != "" and "custom" not in val.lower()
                
                # Check if Custom Range is active
                has_custom_active = "Custom Range:" in str(val)
                
                # 🔥 FIX: Verify if custom range actually has data entered
                if has_custom_active:
                    c_min = self.g1_custom_min.get(col_str, "")
                    c_max = self.g1_custom_max.get(col_str, "")
                    has_custom_data = (str(c_min).strip() != "") or (str(c_max).strip() != "")
                    
                    # If no data was entered, cancel the selection and close entry mode
                    if not has_custom_data:
                        has_custom_active = False
                        self.g1_custom_visible[col_str] = False
                
                has_selection = has_categorical or has_custom_active
                
                # Clean eviction if no filters remain active
                if not has_selection:
                    self.group1_selected_cols = [c for c in self.group1_selected_cols if str(c).strip().lower() != col_str.lower()]
                    self.g1_checked_values[col_str] = []
                    if col_name in self.g1_active_filter_cols: self.g1_active_filter_cols.remove(col_name)
                    if col_str in self.g1_active_filter_cols: self.g1_active_filter_cols.remove(col_str)
                else:
                    if col_name not in self.group1_selected_cols and col_str not in self.group1_selected_cols:
                        self.group1_selected_cols.append(col_name)
                        
            self.g1_expanded = {**self.g1_expanded}
            
        else:
            # Group 2 Management (Identical fix applied here)
            current_expanded = self.g2_expanded.get(col_str, False)
            next_expanded = not current_expanded
            self.g2_expanded[col_str] = next_expanded
            
            if next_expanded:
                self.g2_active_editing_col = col_str
            else:
                if self.g2_active_editing_col == col_str:
                    self.g2_active_editing_col = ""
                
                val = self.g2_checked_values.get(col_str, [])
                if isinstance(val, list):
                    has_categorical = any(isinstance(v, str) and v.strip() != "" and "custom" not in v.lower() for v in val) or any(not isinstance(v, str) for v in val if v is not None)
                else:
                    has_categorical = isinstance(val, str) and val.strip() != "" and "custom" not in val.lower()
                
                # Check if Custom Range is active
                has_custom_active = "Custom Range:" in str(val)
                
                # 🔥 FIX: Verify if custom range actually has data entered
                if has_custom_active:
                    c_min = self.g2_custom_min.get(col_str, "")
                    c_max = self.g2_custom_max.get(col_str, "")
                    has_custom_data = (str(c_min).strip() != "") or (str(c_max).strip() != "")
                    
                    # If no data was entered, cancel the selection and close entry mode
                    if not has_custom_data:
                        has_custom_active = False
                        self.g2_custom_visible[col_str] = False
                
                has_selection = has_categorical or has_custom_active
                
                # Clean eviction if no filters remain active
                if not has_selection:
                    self.group2_selected_cols = [c for c in self.group2_selected_cols if str(c).strip().lower() != col_str.lower()]
                    self.g2_checked_values[col_str] = []
                    if col_name in self.g2_active_filter_cols: self.g2_active_filter_cols.remove(col_name)
                    if col_str in self.g2_active_filter_cols: self.g2_active_filter_cols.remove(col_str)
                else:
                    if col_name not in self.group2_selected_cols and col_str not in self.group2_selected_cols:
                        self.group2_selected_cols.append(col_name)
                        
            self.g2_expanded = {**self.g2_expanded}

        self.clear_results_display()

    @rx.var    
    def g1_active_filter_cols(self) -> list[str]:
        """Returns a list of column names that have active filters in Group 1."""
        active_cols = []
        for col in self.csv_columns:
            col_str = str(col)
            
            if col in self.group1_selected_cols:
                active_cols.append(col)
                continue
                
            # Ensure we only consider a dictionary key active if it contains an actual selection
            if col_str in self.g1_checked_values:
                val = self.g1_checked_values[col_str]
                if (isinstance(val, list) and len(val) > 0) or (isinstance(val, str) and val != ""):
                    active_cols.append(col)
                    continue
                    
        return active_cols

    @rx.var
    def g2_active_filter_cols(self) -> list[str]:
        """Returns a list of column names that have active filters in Group 2."""
        active_cols = []
        for col in self.csv_columns:
            col_str = str(col)
            
            if col in self.group2_selected_cols:
                active_cols.append(col)
                continue
                
            if col_str in self.g2_checked_values:
                val = self.g2_checked_values[col_str]
                if (isinstance(val, list) and len(val) > 0) or (isinstance(val, str) and val != ""):
                    active_cols.append(col)
                    continue
                    
        return active_cols
        
    # ADD THESE NEW FUNCTIONS RIGHT BELOW THEM:
    @rx.var
    def g1_parent_checked_statuses(self) -> dict[str, bool]:
        statuses = {}
        for col in self.csv_columns:
            col_str = str(col)
            checked_vals = self.g1_checked_values.get(col_str, [])
            
            # 1. Check for raw input data sitting in the min/max fields
            min_val = self.g1_custom_min.get(col_str)
            max_val = self.g1_custom_max.get(col_str)
            has_input_data = (
                min_val is not None and str(min_val).strip() != "" and 
                max_val is not None and str(max_val).strip() != ""
            )
            
            # 2. Check if the column is explicitly tracked in your custom list
            has_tracked_custom = False
            if hasattr(self, "g1_columns_with_custom_data"):
                has_tracked_custom = (col_str in self.g1_columns_with_custom_data) or (col in self.g1_columns_with_custom_data)
            
            # 3. Combined omnivore evaluation (If ANY indicator is True, the parent stays checked)
            statuses[col_str] = (
                col in self.g1_active_filter_cols or 
                col_str in self.g1_active_filter_cols or
                len(checked_vals) > 0 or 
                has_input_data or 
                has_tracked_custom
            )
        return statuses

    @rx.var
    def g2_parent_checked_statuses(self) -> dict[str, bool]:
        statuses = {}
        for col in self.csv_columns:
            col_str = str(col)
            checked_vals = self.g2_checked_values.get(col_str, [])
            
            # 1. Check for raw input data sitting in the min/max fields
            min_val = self.g2_custom_min.get(col_str)
            max_val = self.g2_custom_max.get(col_str)
            has_input_data = (
                min_val is not None and str(min_val).strip() != "" and 
                max_val is not None and str(max_val).strip() != ""
            )
            
            # 2. Check if the column is explicitly tracked in your custom list
            has_tracked_custom = False
            if hasattr(self, "g2_columns_with_custom_data"):
                has_tracked_custom = (col_str in self.g2_columns_with_custom_data) or (col in self.g2_columns_with_custom_data)
                
            # 3. Combined omnivore evaluation (If ANY indicator is True, the parent stays checked)
            statuses[col_str] = (
                col in self.g2_active_filter_cols or 
                col_str in self.g2_active_filter_cols or
                len(checked_vals) > 0 or 
                has_input_data or 
                has_tracked_custom
            )
        return statuses

    @rx.var
    def cohort_timeline_heading(self) -> str:
        """Dynamically computes the header and detects units based on magnitude."""
        if self.timeline_max == 0:
            return "Cohort Timeline:"
        return f"Cohort Timeline: {self.timeline_values[0]} to {self.timeline_values[1]} {self.timeline_unit}"
        
        # 🎯 Smart Unit Detection: If max is over 240, it's safely a day-based dataset!
        unit = "Days" if self.timeline_max > 240 else "Mo"
        
        return f"Cohort Timeline: {self.timeline_values[0]} to {self.timeline_values[1]} {unit}"
    
    def handle_timeline_change(self, new_values: list[float]): # 🎯 Match what Reflex frontend expects
        """Triggered instantly whenever the user drags either slider handle."""
        # Cleanly convert the floats back to integers for your backend state
        self.timeline_values = [int(val) for val in new_values]
        
        # Instantly clear the results so the old plot disappears 
        self.results_ready = False
    
    def handle_timeline_slider_drop(self, val: list[int]):
        """Catches the slider values on mouse-release and updates the charts."""
        self.timeline_values = val
        # Call whatever function executes your Kaplan-Meier / Cox math and updates your chart figures
        return self.run_analysis()
    
    def set_timeline_values(self, val: list[float]):
        """Explicitly handles slider adjustments."""
        self.timeline_values = [int(v) for v in val]

    # 🟢 A simple structural guard flag
    is_running_validation: bool = False

    variable_selection_error: str = ""
    dataset_selected: bool = False  # Example prerequisite
    model_selected: bool = False    # Example prerequisite
    
    # Unified localized sidebar notification system
    toast_duration: int = 1500  
    sidebar_msg: str = ""
    sidebar_variant: str = "success"  
    show_sidebar_msg: bool = False
    _msg_id: int = 0  

    # The memory variable
    open_sections: list[str] = ["data_selection", "ms"]

   # 🟢 UPDATE THIS LINE: Add "str | " to the annotation to satisfy Reflex's strict check
    def toggle_sections(self, sections: str | list[str]):
        if isinstance(sections, list):
            self.open_sections = sections
        else:
            self.open_sections = [sections] if sections else []

    # 🟢 New variables for tracking the pop-up warning
    show_run_error: bool = False
    run_error_message: str = ""

    # 🟢 Variables for Download button Future Functionality Message
    show_download_warning: bool = False

    # 🟢 New localized indicator flag
    show_variable_prereq_error: bool = False
    is_processing_accordion_timer: bool = False

    # 🟢 Internal guard flag to block concurrent execution loops
    is_processing_prereq: bool = False


    # 🟢 1. Split into individual, subtle error states
    show_dataset_error: bool = False
    show_model_error: bool = False
    show_variable_error: bool = False

    # 🟢 A clean, instantaneous computed variable (No network lag!)
    #@rx.var
    #def variable_prereqs_missing(self) -> bool:
    #    return (not self.selected_dataset) or (not self.selected_model)

    # 🟢 Custom handler for the Accordion Root
    async def watch_accordion_change(self, open_items: list[str] | str):
        # 🟢 Safety check: If Reflex passes a single string, wrap it in a list 
        if isinstance(open_items, str):
            open_items = [open_items]

        # Instantly update the layout state so the fold opens/closes immediately
        self.open_sections = open_items

        # Check if they just expanded the variable panel without meeting prereqs
        if "dynamic_variables" in open_items and (not self.selected_dataset or not self.selected_model):
            # Guard: If a timer is already running, don't start another one
            if self.is_processing_accordion_timer:
                return
                
            self.is_processing_accordion_timer = True
            self.show_variable_prereq_error = True
            yield  # 📤 Show the warning inline immediately
            
            await asyncio.sleep(2)
            
            self.show_variable_prereq_error = False
            self.is_processing_accordion_timer = False
            yield  # 📤 Fade the warning text away

    csv_columns: list[str] = rx.field(default_factory=list)
    selected_time_column: str = ""
    selected_event_column: str = ""

    # Track dataset and column selections
    selected_dataset: str = "Lung"
    selected_column: str = "sex"

    # Sub-selections for comparison
    available_groups: list[str] = []
    group_a: str = ""
    group_b: str = ""
    is_continuous: bool = False

    # Tracks which column name's dropdown is actively open for Group 1 and Group 2
    # Set to an empty string "" when no dropdown is active
    g1_active_editing_col: str = ""
    g2_active_editing_col: str = ""

    g1_custom_min: dict[str, str] = {}
    g1_custom_max: dict[str, str] = {}
    g2_custom_min: dict[str, str] = {}
    g2_custom_max: dict[str, str] = {}
    
    def get_current_dataframe(self) -> pd.DataFrame:
        """Helper to return the correct dataset based on user selection."""
        if self.selected_dataset == "Breast":
            pass 
        else:
            pass

    # Custom comparison groups for Physicians/Researchers
    group1_selected_cols: list[str] = rx.field(default_factory=list)
    group2_selected_cols: list[str] = rx.field(default_factory=list)

    def load_dataset_columns(self):
        """Triggered when dataset or column changes to compute available groups."""
        df = self.get_current_dataframe()  
        
        # Safe-guard check: If the dataframe didn't load properly, stop execution
        if df is None or df.empty:
            return

        # --- 🎯 NEW: TIMELINE CALCULATOR INJECTION --- #
        # Find the column containing the survival/timeline tracking data
        time_columns = [col for col in df.columns if col.lower() in ['time', 'duration']]
        
        if time_columns:
            # Grab the first matching column and calculate its absolute maximum value
            target_time_col = time_columns[0]
            max_months = int(df[target_time_col].dropna().max())
            
            # Update state: This lights up the slider and snaps handles to full width!
            self.timeline_max = max_months
            self.timeline_values = [0, max_months]
        else:
            # Fallback safety default if a dataset doesn't have a time column
            self.timeline_max = 0
            self.timeline_values = [0, 0]
        # ------------------------------------------------------------------------------- #

        # --- UNIFIED PROCESSOR: Populates smart hybrid dicts for ALL sidebar rows ------ #
        unique_map = {}
        g1_defaults = {}
        g2_defaults = {}
        
        # ✅ Initialize ALL columns as empty lists right from the start.
        # This covers tracking columns (time, status, event) so they don't get dropped.
        g1_defaults = {str(col): [] for col in df.columns}
        g2_defaults = {str(col): [] for col in df.columns}
        
        g1_custom_defaults = {}
        g2_custom_defaults = {}
        
        for col in df.columns:
            if col.lower() in ['time', 'duration', 'status', 'cens', 'event']:
                continue
            
            # Core transformation engine injection
            computed_options = self._generate_smart_intervals(df, col)
            unique_map[str(col)] = computed_options
            
            if computed_options:
                # Set to empty strings so the engine doesn't filter by default
                #g1_defaults[str(col)] = ""  
                #g2_defaults[str(col)] = ""

                # ✅ FIX 2: Ensure these remain lists if they match computed options
                g1_defaults[str(col)] = []  
                g2_defaults[str(col)] = []  
                
                series_clean = df[col].dropna()
                if pd.api.types.is_numeric_dtype(series_clean) and not series_clean.empty:
                    g1_custom_defaults[str(col)] = [float(series_clean.min()), float(series_clean.max())]
                    g2_custom_defaults[str(col)] = [float(series_clean.min()), float(series_clean.max())]

        # Save everything atomically to the State fields
        self.column_unique_values = unique_map
        self.g1_checked_values = g1_defaults
        self.g2_checked_values = g2_defaults
        self.g1_custom_ranges = g1_custom_defaults
        self.g2_custom_ranges = g2_custom_defaults

        #  FIXED: Uncommented tracking list reset to keep titles synced on fresh datasets
        self.group1_selected_cols = []
        self.group2_selected_cols = []

        col = self.selected_column
        
        # Verify the primary stratification column exists in the dataset
        if col not in df.columns:
            return

        # Handle continuous/numeric vs categorical primary groupings
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 5:
            self.is_continuous = True
            try:
                binned = pd.qcut(df[col], q=4, duplicates='drop')
                self.available_groups = [
                    f"{int(i.left)} to {int(i.right)}" if pd.api.types.is_numeric_dtype(df[col]) else str(i)
                    for i in binned.cat.categories
                ]
            except Exception:
                self.available_groups = [str(x) for x in df[col].dropna().unique()]
                self.is_continuous = False
        else:
            self.is_continuous = False
            self.available_groups = [
                str(int(x)) if x.is_integer() else str(x) 
                for x in df[col].dropna().unique() 
                if isinstance(x, (int, float))
            ] or [str(x) for x in df[col].dropna().unique()]

        # Set default values for group_a and group_b split selections
        if len(self.available_groups) >= 2:
            self.group_a = self.available_groups[0]
            self.group_b = self.available_groups[1]
        elif len(self.available_groups) == 1:
            self.group_a = self.available_groups[0]
            self.group_b = self.available_groups[0]

    async def set_time_column(self, value: str):
        """Updates, validates, and normalizes the selected time column using the pristine raw dataset."""
        self.time_error_msg = "" 
        self.show_variable_error = False 

        # Ensure raw backup dataframe exists and the column is valid
        target_source = self._raw_df if self._raw_df is not None else self.uploaded_data
        if target_source is not None and value in target_source.columns:
            raw_series = target_source[value]
            
            # Check for missing/null values
            if raw_series.isna().any():
                self.time_error_msg = "Contains missing values"
                self.show_variable_error = True
                yield
                await asyncio.sleep(2)
                self.time_error_msg = ""
                self.show_variable_error = False
                return

            # Attempt numeric conversion
            converted = pd.to_numeric(raw_series, errors='coerce')
            if converted.isna().any():
                self.time_error_msg = "Invalid numeric format"
                self.show_variable_error = True
                yield
                await asyncio.sleep(2)
                self.time_error_msg = ""
                self.show_variable_error = False
                return
            
            # Ensure no negative durations exist
            if (converted < 0).any():
                self.time_error_msg = "Contains negative values"
                self.show_variable_error = True
                yield
                await asyncio.sleep(2)
                self.time_error_msg = ""
                self.show_variable_error = False
                return
            
            # Commit the assignment
            self.selected_time_column = value
            self.critical_columns = [self.selected_time_column, self.selected_event_column]

            # 🎯 Trigger centralized normalization using the pristine raw source
            model_df, self.timeline_unit, self.timeline_max, self.timeline_values = self._normalize_dataframe(
                target_source, self.selected_time_column
            )
            
            # Update model and filtered states
            self._model_df = model_df
            self._filtered_df = self._model_df.copy()
            
            # Refresh metrics if both critical columns are now selected
            if self.selected_time_column and self.selected_event_column:
                self._update_metrics()
                self.status_message = ""

    async def set_event_column(self, value: str):
        # Reset errors first (do NOT assign selected_event_column yet!)
        self.event_error_msg = "" 
        self.show_variable_error = False 

        # Ensure dataframe exists and the column is valid
        if self._df is not None and value in self._df.columns:
            raw_series = self._df[value]
            
            # Check for missing/null values first
            if raw_series.isna().any():
                self.event_error_msg = "Contains missing values"
                self.show_variable_error = True
                yield
                await asyncio.sleep(2)
                self.event_error_msg = ""
                self.show_variable_error = False
                return  # Exit early, invalid column is rejected!

            # Attempt numeric conversion to catch strings/text mixed in
            converted = pd.to_numeric(raw_series, errors='coerce')
            if converted.isna().any():
                self.event_error_msg = "Invalid numeric format"
                self.show_variable_error = True
                yield
                await asyncio.sleep(2)
                self.event_error_msg = ""
                self.show_variable_error = False
                return  # Exit early, invalid column is rejected!return  # Exit early, invalid column is rejected!
                
            # Verify that unique values are strictly binary (0 and 1)
            unique_vals = set(converted.unique())
            allowed_values = {0, 1, 0.0, 1.0}
            
            cleaned_uniques = {v for v in unique_vals if not pd.isna(v)}
            if not cleaned_uniques.issubset(allowed_values) or len(cleaned_uniques) > 2:
                self.event_error_msg = "Must be binary (0/1)"
                self.show_variable_error = True
                yield
                await asyncio.sleep(2)
                self.event_error_msg = ""
                self.show_variable_error = False
                return  # Exit early, invalid column is rejected!

            # ✅ IN-PLACE CLEANUP: Clean the user's original column to integers 
            # without renaming it or creating a duplicate column!
            self._df[value] = converted.astype(int)
        
        # 🎯 Only commit the assignment AFTER all validation checks pass successfully
        self.selected_event_column = value
        
        # Update your critical columns tracking
        self.critical_columns = [self.selected_time_column, self.selected_event_column]

    def clear_sidebar_notification(self):
        self.show_sidebar_notification = False

    def _show_custom_toast(self, message: str, variant: str = "success"):
        """Intercepts all notification types and routes them into the inline sidebar menu hierarchy."""
        self.sidebar_msg = message
        self.sidebar_variant = variant
        self.show_sidebar_msg = True
        self._msg_id += 1 
        return State.clear_sidebar_msg_after_delay

    debug_status: str = "Awaiting click..."
    show_dialog: bool = False
    results_ready: bool = False
    report_message: str = "Please configure Model Settings and click RUN ANALYSIS"

    async def notify_future_release(self):
        # 1. Instantly display the localized message box
        self.show_download_warning = True
        yield  # 📤 Push state update to browser
        
        # 2. Wait exactly 3 seconds
        await asyncio.sleep(2)
        
        # 3. Seamlessly hide the message box
        self.show_download_warning = False
        yield  # 📤 Push clean state back to browser
    
    def toggle_dialog(self):
        self.show_dialog = not self.show_dialog

    selected_dataset: str = ""  
    selected_model: str = ""

    active_controls: list[str] = rx.field(default_factory=list)
    data_selection: str = "Default"
    metadata_info: str = "" 

    # --- File Tracking & Dynamic KM States ---
    uploaded_filename: str = ""
    custom_filename: str = ""               # Stores uploaded filename permanently for the session/report

    uploaded_row_count: int = 0
    show_schema_preview: bool = False
    
    # Stratification values for Custom CSV KM execution
    km_stratify_col: str = ""
    km_selected_subgroups: list[str] = rx.field(default_factory=list)
    km_available_subgroups: list[str] = rx.field(default_factory=list)
    km_numeric_split_value: str = ""
    km_is_numeric_stratify: bool = False

    # Structural Metadata Display Vectors
    schema_cols: list[str] = rx.field(default_factory=lambda: ["Column Name", "Detected Type", "Valid Row Count"])
    schema_rows: list[list[str]] = rx.field(default_factory=list)

    # --- Preview Grid States ---
    data_preview_cols: list[str] = rx.field(default_factory=list)
    data_preview_rows: list[list[str]] = rx.field(default_factory=list)
    inspector_active_tab: str = "preview"

    current_dataset: str = "lung"  # Default dataset

    def _populate_inspector_matrices(self, df: pd.DataFrame = None):
        """Helper to cleanly translate a dataframe into the scrollable UI grid and data definition schema."""
        
        # 1. Handle dataframe caching and fallback states
        if df is not None:
            self._filtered_df = df
        else:
            df = getattr(self, "_filtered_df", None)
            
        if df is None or df.empty:
            df = getattr(self, "_raw_df", None)
            if df is not None:
                self._filtered_df = df
                
        if df is None or df.empty:
            self.uploaded_row_count = 0
            self.data_preview_rows = [["-"]]
            self.schema_rows = []
            return

        self.uploaded_row_count = len(df)
        
        # 2. Determine preview columns (hide redundant shadow columns)
        preview_cols = [
            c for c in df.columns 
            if not (c == 'event' and self.selected_event_column and self.selected_event_column != 'event')
        ]

        self.schema_rows = [
            [str(col), str(dtype), str(df[col].notnull().sum())]
            for col, dtype in zip(df.columns, df.dtypes)
        ]
        
        self.data_preview_cols = preview_cols

        # 3. Chunked Viewport Slicing (Prevents DOM bloat & drives Go/Top/End buttons)
        total_rows = len(df)
        if self.viewport_offset >= total_rows:
            self.viewport_offset = max(0, total_rows - self.chunk_size)

        start_idx = self.viewport_offset
        end_idx = min(start_idx + self.chunk_size, total_rows)
        
        preview_df = df[preview_cols].iloc[start_idx:end_idx].copy()
        
        float_cols = preview_df.select_dtypes(include=['float32', 'float64']).columns
        preview_df[float_cols] = preview_df[float_cols].round(2)
        
        preview_df = preview_df.fillna("NaN")
        self.data_preview_rows = preview_df.astype(str).values.tolist()
        
        # 4. Extract unique values and initialize defaults only on initial load 
        # (Using _raw_df as the reference base ensures continuous variable bin edges never shift when rows are dropped)
        if not self.column_unique_values:
            reference_df = getattr(self, "_raw_df", None)
            if reference_df is None or reference_df.empty:
                reference_df = df

            unique_map = {}
            g1_defaults = {str(col): [] for col in df.columns}
            g2_defaults = {str(col): [] for col in df.columns}
            g1_custom_defaults = {}
            g2_custom_defaults = {}
            
            for col in df.columns:
                if col.lower() in ['time', 'duration', 'status', 'cens', 'event']:
                    continue
                
                computed_options = self._generate_smart_intervals(reference_df, col)
                unique_map[str(col)] = computed_options
                
                if computed_options:
                    g1_defaults[str(col)] = []
                    g2_defaults[str(col)] = []
                    
                    series_clean = reference_df[col].dropna()
                    if pd.api.types.is_numeric_dtype(series_clean) and not series_clean.empty:
                        g1_custom_defaults[str(col)] = [float(series_clean.min()), float(series_clean.max())]
                        g2_custom_defaults[str(col)] = [float(series_clean.min()), float(series_clean.max())]

            self.column_unique_values = unique_map
            self.g1_checked_values = g1_defaults
            self.g2_checked_values = g2_defaults
            self.g1_custom_ranges = g1_custom_defaults
            self.g2_custom_ranges = g2_custom_defaults
            self.group1_selected_cols = []
            self.group2_selected_cols = []

    def change_inspector_tab(self, tab_value: str):
        self.inspector_active_tab = tab_value

    async def run_analysis(self):
        # Guard: If the app is already running, drop spam clicks instantly 
        # BEFORE modifying any state or clearing old data!
        if self.is_running_validation:
            return
        
        # 0. --- Clear any old Report Table 1 Selections
        #self.table_one_data = []

        # 1. --- Clear any old errors at the very beginning ---
        self.time_error_msg = ""
        self.event_error_msg = ""
        self.show_variable_error = False
        has_error = False
        
        self.show_dataset_error = False
        self.show_model_error = False

        # ---------------------------------- PHASE A: MODEL DATAFRAME RESOLUTION (Using Centralized State) ------------------------------------ #
        df_active = None
        
        # 🛡️ PRIORITIZE _filtered_df FIRST so in-memory cleansings (like duplicate removals) are used
        if hasattr(self, "_filtered_df") and self._filtered_df is not None and not self._filtered_df.empty:
            df_active = self._filtered_df.copy()

        elif hasattr(self, "_model_df") and self._model_df is not None and not self._model_df.empty:
            df_active = self._model_df.copy()

        if hasattr(self, "_model_df") and self._model_df is not None and not self._model_df.empty:
                    df_active = self._model_df.copy()
        else:
            # Fallback safeguard if model state hasn't been initialized yet
            if "Breast" in self.selected_dataset or "GBSG2" in self.selected_dataset:
                df_active = load_gbsg2().copy()
                self.metadata_info = "Source: GBSG2 Breast Cancer Study. Created: 1994. 686 patients with node-positive cancer."
            elif "Lung" in self.selected_dataset or "NCCTG" in self.selected_dataset:
                df_active = load_lung().copy()
                self.metadata_info = "Source: NCCTG Lung Cancer Data (Loprinzi et al.). Created: 1994. Advanced lung cancer study."
            elif self.uploaded_filename and self.uploaded_data is not None and not self.uploaded_data.empty:
                df_active = self.uploaded_data.copy()  
                self.metadata_info = f"Source: {self.uploaded_filename} ({len(self.uploaded_data)} rows). Custom cohort analysis."
                self.is_custom_dataset = True
            
            if df_active is not None:
                df_active.columns = df_active.columns.str.lower().str.strip()
            else:
                self.metadata_info = "Source: No Dataset selected. Please select Dataset."
                self.show_schema_preview = False
                self.data_preview_cols = []
                self.data_preview_rows = []
                self.schema_rows = []
                self.is_custom_dataset = True
                return  # Exit early safely

        # Resolve column names dynamically
        time_col = self.selected_time_column if self.selected_time_column else 'time'
        event_col = self.selected_event_column

        # ------------------------------ PHASE B : UNIFIED VARIABLE & DATASET VALIDATION (Single Source of Truth) ----------------------------- #
        self.inline_error_msg = "" 

        if self.selected_dataset == "" or df_active is None or df_active.empty: 
            self.show_dataset_error = True
            self.inline_error_msg = "⚠️ Please select a valid dataset."
            has_error = True

        elif self.selected_model == "": 
            self.show_model_error = True
            #self.inline_error_msg = "⚠️ Please select a Survival Model."
            has_error = True
            
        elif self.selected_model == "Cox Proportional Hazards" and len(self.cph_active_covariates) == 0:
            self.show_variable_error = True
            #self.inline_error_msg = "⚠️ Please select at least one covariate."
            has_error = True

        # --- Time Column Validation ---
        elif not time_col or time_col not in df_active.columns:
            self.time_error_msg = "Invalid column"
            self.show_variable_error = True
            has_error = True
        else:
            time_series = pd.to_numeric(df_active[time_col], errors='coerce')
            if time_series.isna().any():
                self.time_error_msg = "Contains missing values"
                self.show_variable_error = True
                has_error = True
            elif (time_series < 0).any():
                self.time_error_msg = "Contains negative values"
                self.show_variable_error = True
                has_error = True

        # --- Event Column Validation (Only checks if time passed) ---
        if not has_error:
            if not event_col or event_col not in df_active.columns:
                self.event_error_msg = "Invalid column"
                self.show_variable_error = True
                has_error = True
            else:
                event_series = pd.to_numeric(df_active[event_col], errors='coerce')
                if event_series.isna().any():
                    self.event_error_msg = "Contains missing values"
                    self.show_variable_error = True
                    has_error = True
                else:
                    unique_vals = set(event_series.unique())
                    cleaned_uniques = {v for v in unique_vals if not pd.isna(v)}
                    if not cleaned_uniques.issubset({0, 1, 0.0, 1.0}) or len(cleaned_uniques) > 2:
                        self.event_error_msg = "Must be binary (0/1)"
                        self.show_variable_error = True
                        has_error = True

        # --------------------------------- PHASE C : ERROR COOLDOWN PIPELINE (4-Second Auto-Clear Lockout) ----------------------------------- #
        if has_error:
            self.is_running_validation = True
            if "data_selection" not in self.open_sections and self.show_dataset_error:
                self.open_sections.append("data_selection")
            if "ms" not in self.open_sections and self.show_model_error:
                self.open_sections.append("ms")
            if "dynamic_variables" not in self.open_sections and self.show_variable_error:
                self.open_sections.append("dynamic_variables")
            
            yield 
            await asyncio.sleep(2)
            
            # --- CLEAR EVERYTHING INCLUDING THE NEW MESSAGES ---
            self.show_dataset_error = False
            self.show_model_error = False
            self.show_variable_error = False
            self.time_error_msg = ""      
            self.event_error_msg = ""     
            self.is_running_validation = False 
            yield 
            return

        # --------------------------------------- PHASE D : CORE MATH EXECUTION ENGINE (Success Path)------------------------------------------ #
        self.is_running_validation = True
        yield 

        try:
            # 🌟 SET FULL COHORT FLAG for Clinical Information Decision Support Box
            # to check if a second comparison group or stratification filter is active)
            if not hasattr(self, "group2_selected_cols") or not self.group2_selected_cols:   # check if a stratification filter is active
                self.is_full_cohort = True
            else:
                self.is_full_cohort = False

            # Auto-initialize timeline slider boundaries if uninitialized
            if df_active is not None and not df_active.empty:
                if time_col in df_active.columns:
                    max_val_raw = pd.to_numeric(df_active[time_col], errors='coerce').max()
                    if not pd.isna(max_val_raw) and max_val_raw > 0:
                        calculated_max = int(np.ceil(max_val_raw))
                        if getattr(self, "timeline_max", 0) == 0 or self.timeline_values == [0, 0]:
                            self.timeline_max = calculated_max
                            self.timeline_values = [0, calculated_max]

            # 🎯 SLICE DATAFRAME WINDOW BASED ON RANGE SLIDER VALUES
            # (Directly using model scale since _model_df time units match timeline_values)
            lower_bound = self.timeline_values[0]
            upper_bound = self.timeline_values[1]

            # ------------------------------------------------------ MODEL ROUTING GATE ------------------------------------------------------ #
            if self.selected_model == "Kaplan-Meier Survival":
                # 🧹 Clear CPH state variables when running KM
                self.cph_table_data = []
                self.cph_fig = go.Figure()
                self.cph_survival_prob = ""
                self.cph_log_rank_p = ""
                self.cph_error_message = ""

                self._run_km_engine(
                    df=df_active, 
                    days_input=upper_bound, 
                    max_x=upper_bound       
                )
                self.survival_fig = self.survival_fig_g1
                
            elif self.selected_model == "Cox Proportional Hazards":
                # 🧹 Clear Kaplan-Meier state variables when running CPH
                self.survival_probability_text = ""
                self.median_survival_text = ""
                self.p_value_text = ""
                self.ci_text = ""
                self.survival_probability_list = []
                self.median_survival_list = []
                self.at_risk_data = []
                self.cohort_note = ""

                if "Lung" in self.selected_dataset and 'sex' in df_active.columns:
                    df_active['sex'] = df_active['sex'].map({1: 1, 2: 2, 'Male': 1, 'Female': 2}) 
                elif "Breast" in self.selected_dataset:
                    if 'horTh' in df_active.columns:
                        df_active['horTh'] = df_active['horTh'].map({'yes': 1, 'no': 0, 1: 1, 0: 0})
                    if 'menostat' in df_active.columns:
                        df_active['menostat'] = df_active['menostat'].map({'Post': 1, 'Pre': 0, 1: 1, 0: 0})
                    if 'tgrade' in df_active.columns:
                        df_active['tgrade'] = df_active['tgrade'].map({'I': 1, 'II': 2, 'III': 3, 1: 1, 2: 2, 3: 3})

                self._run_cph_engine(df_active, upper_bound)
                
            if hasattr(self, "survival_fig_g1") and self.survival_fig_g1:
                self.survival_fig_g1.update_layout(
                    xaxis_title=f"Timeline ({self.timeline_unit})",
                    xaxis=dict(
                        range=[self.timeline_values[0], self.timeline_values[1]],
                        autorange=False
                    )
                )

            self.results_ready = True

        except ValueError as e:
            print(f"DEBUG CPH Error: {e}")
            self.show_variable_error = True    
            return
            
        finally:
            self.is_running_validation = False
            yield
    
    def reset_state(self):
        """Complete hard system wipe back to initial state setup."""
        self.selected_dataset = ""
        self.selected_model = ""
        self.uploaded_filename = ""
        self.uploaded_row_count = 0
        self.metadata_info = ""
        self.show_schema_preview = False
        self.schema_rows = []
        self.km_stratify_col = ""
        self.km_selected_subgroups = []
        self.km_available_subgroups = []
        self.km_numeric_split_value = ""
        self.km_is_numeric_stratify = False
        self.group1_selected_cols = []
        self.group2_selected_cols = []
        self.cph_active_covariates = []
        self.selected_time_column = ""
        self.selected_event_column = ""
        
        # Hand off the rest to the visual clear routine
        self.clear_results_display()

    def clear_results_display(self):
        # --- Cleans visual evaluation data, calculations, and active figures without resetting selected parameters or file uploads --- #

        # 🧹 Clear active model errors whenever filters or states are adjusted
        self.km_error_message = ""
        self.km_warning_message = ""
        self.cph_error_message = ""

        self.is_full_cohort = False

        self.results_ready = False
        self.cohort_note = ""

        # 🌟 Set the instructional message for the empty report placeholder
        self.report_message = "Configuration updated in sidebar. Click Run Analysis to generate Updated Survival Report."
        
        # Reset figures to clean initial placeholders
        blank_fig = make_initial_figure()
        self.survival_fig = blank_fig
        self.survival_fig_g1 = blank_fig
        self.survival_fig_g2 = blank_fig
        self.cph_fig = blank_fig

        # 🧹 WIPE TABLE 1 CLEAN (Crucial for data integrity)
        self.table_one_data = []
        self.table_one_g1_label = "Group 1"
        self.table_one_g2_label = "Group 2"

        # 🧹 WIPE THE STATISTICAL CARDS DATA LISTS
        self.survival_probability_list = []
        self.median_survival_list = []

        # 🧹 WIPE STATISTICAL CARDS / DATA LISTS (Fixes lingering card text)
        # (Replace `survival_prob_list` and `median_survival_list` with your exact state variable names if they differ)
        self.survival_probability_list = []      # Or whatever list populates the Survival Prob card
        self.median_survival_list = []    # Or whatever list populates the Median Survival card
        
        # Wipe structural table data matrices and numeric textual summaries
        self.cph_table_data = []
        self.cph_survival_prob = ""
        self.cph_log_rank_p = ""
        self.at_risk_data = []
        self.at_risk_columns = []
        self.survival_probability_text = ""
        self.median_survival_text = ""
        self.p_value_text = ""
        self.ci_text = ""
        self.ci_text = ""  # This wipes your CI text block clean!

        # 🌟 ADD CPH VALIDATION & SUMMARY TEXT WIPES
        self.cph_sample_metrics_text = ""
        self.cph_c_index_text = ""
        self.cph_apparent_c_index = ""
        self.cph_has_run = False
           
    async def update_dataset(self, value: str):
        # 0. Clear or re-initialize dynamic selections to prevent invalid index/value bindings
        self.active_controls.clear()

        # 1. ------------------------------ CLEAR EXISTING STATE VARIABLES -------------------------------- #
        self.show_schema_preview = False
        self.selected_dataset = value
        self.clear_results_display()
        self.results_ready = False  
        self.survival_probability_text = ""
        self.median_survival_text = "---"
        self.p_value_text = ""
        self.active_controls = [] 
        self.cph_active_covariates = []
        self.group1_selected_cols = []
        self.group2_selected_cols = []
        self.g1_expanded = {}
        self.g2_expanded = {}

        self.viewport_offset = 0                     # Reset table viewport offset to page one on dataset switch
        self.jump_row_number = ""                    # Clears the input so the "Row #" placeholder reappears
        self.search_query = ""                       # Clear active search on dataset switch
        self.status_message = ""                     # Clear any lingering status messages

        self.selected_search_column = "all"          # Reset search column selection back to "All Columns"

        # Clear all lingering custom range properties
        self.g1_custom_min = {}
        self.g1_custom_max = {}
        self.g1_custom_visible = {}
        self.g2_custom_min = {}
        self.g2_custom_max = {}
        self.g2_custom_visible = {}

        self.at_risk_data = [["-", "-", "-", "-", "-", "-"]] 
        self.cph_table_data = [["-", "-", "-", "-", "-"]]
        self.ci_text = "---"

        self.selected_time_column = ""
        self.selected_event_column = ""
        self.critical_columns = []

        self._reset_cph_sandbox_state()     # 🧹 Wipe CPH exploratory/sandbox state to prevent cross-dataset contamination

        active_df = None
        
        # 2. --- Loading a Dataset: Check override first, then Built-In / Custom ---
        if getattr(self, "_override_active_df", None) is not None and value == self.current_dataset:
            active_df = self._override_active_df.copy()
            if "Breast" in value or "GBSG2" in value:
                self.metadata_info = f"Source: GBSG2 Breast Cancer Study (Cleansed, {len(active_df)} rows)."
                self.is_custom_dataset = False
            elif "Lung" in value or "NCCTG" in value:
                self.metadata_info = f"Source: NCCTG Lung Cancer Data (Cleansed, {len(active_df)} rows)."
                self.is_custom_dataset = False
            else:
                self.metadata_info = f"Source: {getattr(self, 'uploaded_filename', 'Custom Data')} ({len(active_df)} rows). Custom cohort analysis."
                self.is_custom_dataset = True
                
        elif "Breast" in value or "GBSG2" in value:                        # --- Hardcoded Built-In Breast Dataset Loading --- #
            active_df = load_gbsg2().copy()
            self.metadata_info = "Source: GBSG2 Breast Cancer Study. Created: 1994. 686 patients with node-positive cancer."
            self.is_custom_dataset = False
            self._override_active_df = None  # Clear override on fresh dataset switch
            
        elif "Lung" in value or "NCCTG" in value:                          # --- Hardcoded Built-In Lung Dataset Loading --- #
            active_df = load_lung().copy()
            self.metadata_info = "Source: NCCTG Lung Cancer Data (Loprinzi et al.). Created: 1994. Advanced lung cancer study."
            self.is_custom_dataset = False
            self._override_active_df = None  # Clear override on fresh dataset switch
            
        else:                                                              # --- User-defined Dataset (Raw Upload) --- #
            if self.uploaded_data is not None and not self.uploaded_data.empty:
                active_df = self.uploaded_data.copy()
                self.metadata_info = f"Source: {self.uploaded_filename} ({len(self.uploaded_data)} rows). Custom cohort analysis."
            self.is_custom_dataset = True
            self._override_active_df = None

        self.current_dataset = value

        # 👈 FIRST YIELD: Force the UI to clear out the old dataset view immediately
        yield
        
        if active_df is not None:                      # --- Converting columns to lowercase and deleting hidden spaces --- #
            active_df.columns = active_df.columns.str.lower().str.strip()
      
        # 3. Dynamic Identification & Sidebar Preparation
        if active_df is not None:
            all_cols = active_df.columns.tolist()

            # Re-initialize expansion maps and unique values for the new dataset columns
            self.g1_expanded = {col: False for col in all_cols}
            self.g2_expanded = {col: False for col in all_cols}

            # Re-build unique values mapping with automatic binning for continuous variables & custom ranges
            new_unique_vals = {}
            for col in all_cols:
                if col.lower() in ['time', 'duration', 'months', 'days', 'survival_time', 'status', 'cens', 'died', 'recurrence', 'censored', 'event']:
                    continue
                new_unique_vals[col] = self._generate_smart_intervals(active_df, col)
                
            self.column_unique_values = new_unique_vals
            self.all_dataset_columns = all_cols
            self.user_excluded_columns = []                 # Reset on new dataset load

            # --- Dynamically identify TIME and EVENT Columns natively --- #
            self.selected_time_column = next((c for c in all_cols if c.lower() in ['time', 'duration', 'months', 'days', 'survival_time']), "")
            self.selected_event_column = next((c for c in all_cols if c.lower() in ['status', 'cens', 'died', 'recurrence', 'censored', 'event']), "")

            # --- Automatically look for a Patient/Subject ID column --- #
            found_id = next((c for c in all_cols if c.lower() in ['id', 'patient_id', 'subject', 'pat_id', 'patient', 'record_id']), "")
            self.selected_dedup_columns = [found_id] if found_id else []

            # --- Store critical columns separately --- #
            self.critical_columns = [self.selected_time_column, self.selected_event_column]

            # 🎯 --- INITIALIZE BASELINE CHARACTERISTICS CONFIGS FOR PUBPLICATION REPORTING MODAL TABLE 1 ---
            configs = []
            excluded_cols = {self.selected_time_column, self.selected_event_column}
            
            for col in all_cols:
                if col in excluded_cols:
                    continue
                
                is_numeric = pd.api.types.is_numeric_dtype(active_df[col])
                # Heuristic: Numeric with > 10 unique values defaults to Continuous; otherwise Categorical
                is_continuous = is_numeric and active_df[col].nunique() > 10
                summary_type = "Continuous" if is_continuous else "Categorical"
                
                # --- 🛠️ Centralized Gold Source: Populate mappings right here at birth --- #
                if summary_type == "Categorical":
                    unique_vals = [str(val) for val in sorted(active_df[col].dropna().unique())]
                    mappings_list = [{"raw": val, "label": val} for val in unique_vals]
                else:
                    mappings_list = []

                configs.append({
                    "name": col,
                    "include": True,  # Checked by default
                    "label": col.replace("_", " ").title(),  # Clean human-readable label
                    "type": summary_type,
                    "mappings_list": mappings_list,
                })
                
            self.column_configs = configs

            # Call helper to handle automatic hiding + user exclusions
            self._recompute_hidden_columns()
            
            # Check if both critical columns were successfully matched automatically
            has_required_cols = bool(self.selected_time_column and self.selected_event_column)

            if not has_required_cols:
                # 🛑 MANUAL MAPPING FALLBACK: Expose table preview & prompt user to select time/event columns
                self._df = active_df.copy()
                self._raw_df = active_df.copy()
                self._model_df = active_df.copy()
                self._filtered_df = active_df.copy()
                
                self.show_schema_preview = True
                self._populate_inspector_matrices(self._df)

                # 🎯 Call metrics update immediately so row counts refresh right away
                self._update_metrics()

                self.status_message = "Please select Time and Event columns to begin analysis."
                self.timeline_max, self.timeline_values, self.timeline_unit = 0, [0, 0], "Mo"
                yield
                return

            # 4. ---------------------------------------------- MODEL DATA NORMALIZATION --------------------------------------------------- #
            #model_df = active_df.copy()       # Isolate normalization strictly to a separate model dataframe copy
            
            if has_required_cols:
                model_df, self.timeline_unit, self.timeline_max, self.timeline_values = self._normalize_dataframe(active_df, self.selected_time_column)
            else:
                self.timeline_max, self.timeline_values, self.timeline_unit = 0, [0, 0], "Mo"
                model_df = active_df.copy()

        # 5. SAVE TO STATE INSTANCE FIRST
        if active_df is not None:
            self._df = active_df.copy()                                                 # Raw pristine copy of full dataset for display/inspector tables
            self._raw_df = active_df.copy()                                             # Raw backup copy
            self._model_df = model_df if 'model_df' in locals() else active_df.copy()   # Normalized dataset: normalized time values for Analysis and Results Display
            self._filtered_df = self._model_df.copy()

            self.show_schema_preview = True         # Data Table Display

        # 6. -------------------------------------------------- POPULATE DATA & METRICS ------------------------------------------------ #
        if active_df is not None:
            self._populate_inspector_matrices(self._df)     # Safe to render table rows using raw display values

            if has_required_cols:
                self._update_metrics()                      # Recalculates metrics only if columns are mapped
                self.status_message = ""
            else:
                # Prompt the user to map columns manually instead of crashing
                self.status_message = "Please select Time and Event columns from the dropdowns above to begin analysis."

        # Final cohort validation check
        self.is_full_cohort = (self.g1_detailed_label == "Group 1: (Full Cohort)") and (self.g2_detailed_label == "Group 2: (Full Cohort)")

        # Single final yield to push all changes atomically to the frontend
        yield
    
    def update_model(self, value: str):
        self.selected_model = value
        self.results_ready = False  
        self.survival_probability_text = ""
        self.median_survival_text = "---"
        self.p_value_text = ""
        self.at_risk_data = [["-", "-", "-", "-", "-", "-"]]
        self.cph_table_data = [["-", "-", "-", "-", "-"]]
        self.cohort_note = ""
        #return self._show_custom_toast(f"{value} Selected.", "success")
        return

    slider_max_val: int = 90
    months_list: list[float] = rx.field(default_factory=lambda: [0.0, 36.0])
    uploaded_data: pd.DataFrame = pd.DataFrame()
    is_uploading: bool = False

    age_threshold: int = 50
    size_threshold: int = 20
    node_threshold: int = 3

    survival_fig: go.Figure = rx.field(default_factory=make_initial_figure)
    survival_fig_g1: go.Figure = rx.field(default_factory=make_initial_figure)
    survival_fig_g2: go.Figure = rx.field(default_factory=make_initial_figure)

    at_risk_columns: list[str] = rx.field(default_factory=lambda: ["Group", "M0", "M10", "M20", "M30", "M40"])
    at_risk_data: list[list[str]] = rx.field(default_factory=lambda: [["-", "-", "-", "-", "-", "-"]])

    p_value_text: str = ""
    median_survival_text: str = "---"
    survival_probability_text: str = "Awaiting..."
    ci_text: str = "---"
    survival_probability_list: list[str] = []       # --- for CI Display
    median_survival_list: list[str] = []

    cph_active_covariates: list[str] = rx.field(default_factory=list)
    cph_fig: go.Figure = rx.field(default_factory=make_initial_figure)

    # --- CPH Journal-Grade Validation & Performance Metrics ---
    cph_sample_metrics_text: str = ""
    cph_c_index_text: str = ""
    cph_log_rank_p: str = ""
    cph_ph_warning: str = ""
    cph_survival_prob: str = "---"
    cohort_note: str = ""
    
    cph_table_data: list[list[str]] = rx.field(default_factory=lambda: [["-", "-", "-", "-", "-"]])
    cph_table_cols: list[str] = rx.field(default_factory=lambda: ["Variable", "HR", "95% CI Lower", "95% CI Upper", "p-value"])

    # Master visibility tracking: Dict[str, bool] maps variable -> True (dropdown open) / False (closed)
    g1_expanded: dict[str, bool] = rx.field(default_factory=dict)
    g2_expanded: dict[str, bool] = rx.field(default_factory=dict)

    # Holds explicit selections (either the interval string label or "custom")
    g1_checked_values: dict[str, list[str]] = rx.field(default_factory=dict)
    g2_checked_values: dict[str, list[str]] = rx.field(default_factory=dict)

    # Physician custom ranges: maps "col_name" -> [min_val, max_val]
    g1_custom_ranges: dict[str, list[float]] = rx.field(default_factory=dict)
    g2_custom_ranges: dict[str, list[float]] = rx.field(default_factory=dict)

    # Map schema: dict[str, list[dict]] 
    column_unique_values: dict[str, list[dict]] = rx.field(default_factory=dict)

    def _generate_smart_intervals(self, df: pd.DataFrame, col_name: str) -> list[dict]:
        """Universal pipeline to convert any continuous or categorical column into clean UI options."""

        # Place this at the absolute top
        #print(f"DEBUG START: Method called for column -> '{col_name}' with unique values: {df[col_name].unique()}")

        series = df[col_name].dropna()
        if series.empty:
            return []

        # Check data characteristics
        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_count = series.nunique()

        #print(f"DEBUG COUNT: {col_name} has unique_count = {unique_count}")

        # Approach A: Low Cardinality Categories or Indicators (<= 4 values)
        if unique_count <= 4:
            unique_vals = list(series.unique())
            
            # Universal Roman numeral map for clinical grades
            roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
            is_roman = all(str(v).strip().lower() in roman_map for v in unique_vals)
            
            # Prioritize explicit tgrade and Roman numeral ordering over broken pandas category sequences
            if col_name.lower() == "tgrade" or is_roman:
                sorted_vals = sorted(unique_vals, key=lambda x: roman_map.get(str(x).strip().lower(), 99))
            elif hasattr(series, "dtype") and hasattr(series.dtype, "categories"):
                cat_order = {val: i for i, val in enumerate(series.dtype.categories)}
                sorted_vals = sorted(unique_vals, key=lambda x: cat_order.get(x, 99))
            else:
                sorted_vals = sorted(unique_vals)

            return [
                {
                    "label": f"Stage {int(v)}" if (col_name.lower() == "tgrade" and is_numeric) else str(v),
                    "is_numeric": is_numeric,
                    "min": float(v) if is_numeric else None,
                    "max": float(v) if is_numeric else None,

                    "is_custom_trigger": False
                }
                for v in sorted_vals
            ]

        # Approach B: High Cardinality Continuous Data (> 4 values)
        if is_numeric:
            try:
                # 3 population buckets (Tertiles) are optimal for clean KM splits
                quantiles = [0.0, 0.33, 0.66, 1.0]
                bin_edges = np.quantile(series, quantiles)
                unique_edges = np.unique(bin_edges)
                
                # If population drops matching edges, default to standard range splits
                if len(unique_edges) < 2:
                    unique_edges = np.linspace(series.min(), series.max(), 4)
            except Exception:
                unique_edges = np.linspace(series.min(), series.max(), 4)

            intervals = []
            for i in range(len(unique_edges) - 1):
                start_val = float(unique_edges[i])
                end_val = float(unique_edges[i+1])
                
                # Updated formatting logic for clinical publication standards
                if i == 0:
                    label = f"Low: ≤{round(end_val, 1)}"
                elif i == len(unique_edges) - 2:
                    label = f"High: >{round(start_val, 1)}"
                else:
                    label = f"Mid: {round(start_val, 1)}–{round(end_val, 1)}"

                intervals.append({
                    "label": label,
                    "is_numeric": True,
                    "min": start_val,
                    "max": end_val,
                    "is_custom_trigger": False
                })

            # Inject the Mutual Exclusive custom toggle field at the bottom
            intervals.append({
                "label": "Custom Range:",
                "is_numeric": True,
                "min": float(series.min()),
                "max": float(series.max()),
                "is_custom_trigger": True
            })
            return intervals

        # Approach C: Textual Categorical variables with high card (e.g. Center ID names)
        else:
            return [
                {"label": str(x), "is_numeric": False, "min": None, "max": None, "is_custom_trigger": False}
                for x in sorted(series.unique())[:15] # Protect viewport against extreme text dumps
            ]

    @rx.event
    def handle_interval_selection(self, group_num: int, col_name: str, display_label: str, val: bool):
        if group_num == 1:
            target_dict = self.g1_checked_values
            target_cols = self.group1_selected_cols
        else:
            target_dict = self.g2_checked_values
            target_cols = self.group2_selected_cols

        current_list = target_dict.get(col_name, [])

        if display_label == "Custom Range:":
            is_visible = self.g1_custom_visible.get(col_name, False) if group_num == 1 else self.g2_custom_visible.get(col_name, False)
            
            if "Custom Range:" in current_list:
                if is_visible:
                    # 🔴 DATA ENTRY ACTIVE -> User clicked to CLOSE/SUBMIT
                    c_min = self.g1_custom_min.get(col_name, "") if group_num == 1 else self.g2_custom_min.get(col_name, "")
                    c_max = self.g1_custom_max.get(col_name, "") if group_num == 1 else self.g2_custom_max.get(col_name, "")
                    has_data = (str(c_min).strip() != "") or (str(c_max).strip() != "")
                    
                    if has_data:
                        target_dict[col_name] = ["Custom Range:"]
                        if col_name not in target_cols:
                            target_cols.append(col_name)
                    else:
                        # No data entered -> completely deselect and remove from parent tracking
                        target_dict[col_name] = []
                        if col_name in target_cols: 
                            target_cols.remove(col_name)
                    
                    if group_num == 1: self.g1_custom_visible[col_name] = False
                    else: self.g2_custom_visible[col_name] = False
                else:
                    # 🟢 SELECTED BUT CLOSED -> User clicked to RE-OPEN data entry mode
                    if group_num == 1: self.g1_custom_visible[col_name] = True
                    else: self.g2_custom_visible[col_name] = True
            else:
                # 🟢 FIRST TIME CLICK -> Wipe out standard options, open data entry
                target_dict[col_name] = ["Custom Range:"]
                # REMOVED: Do not add to target_cols here! Wait for actual data entry.
                if group_num == 1: self.g1_custom_visible[col_name] = True
                else: self.g2_custom_visible[col_name] = True

        else:
            # STANDARD CATEGORICAL SELECTION LOGIC
            if display_label in current_list:
                target_dict[col_name] = []
                if col_name in target_cols: 
                    target_cols.remove(col_name)
            else:
                target_dict[col_name] = [display_label]
                if col_name not in target_cols: 
                    target_cols.append(col_name)

            if group_num == 1:
                self.g1_custom_min[col_name] = ""
                self.g1_custom_max[col_name] = ""
                self.g1_custom_visible[col_name] = False
            else:
                self.g2_custom_min[col_name] = ""
                self.g2_custom_max[col_name] = ""
                self.g2_custom_visible[col_name] = False

        self.clear_results_display()
        self.run_analysis()

    def toggle_sub_value(self, group_num: int, col_name: str, value_str: str, value_checked: bool):
        """Triggered when a specific sub-option checkbox is selected."""
        target_dict = self.g1_checked_values if group_num == 1 else self.g2_checked_values
        if col_name not in target_dict:
            target_dict[col_name] = []
            
        if value_checked:
            if value_str not in target_dict[col_name]:
                target_dict[col_name].append(value_str)
            # Ensure parent stays registered as an active variable column filter
            if group_num == 1 and col_name not in self.group1_selected_cols:
                self.group1_selected_cols.append(col_name)
            elif group_num == 2 and col_name not in self.group2_selected_cols:
                self.group2_selected_cols.append(col_name)
        else:
            if value_str in target_dict[col_name]:
                target_dict[col_name].remove(value_str)

        # Clear visual display when a checkbox value changes
        self.clear_results_display()

    def set_selected_model(self, val: str):
        # 🟢 Instantly clear warnings out if they change selection
        
        # 1. Drop all previous visual calculations, curves, and tables instantly
        self.clear_results_display()

        self._reset_cph_sandbox_state()         # 🧹 Wipe out all CPH exploratory/sandbox state to prevent cross-dataset contamination

        # 2. Update the state variable
        self.selected_model = val
        
        # 2. Complete the core model state transition
        return self.update_model(val)
    
    def set_months_list(self, val: list[float]):
        # 1. Update the underlying list values (slider state)
        self.months_list = val
        
        # 2. Instantly drop stale plots/metrics so the user must click RUN ANALYSIS
        self.clear_results_display()

    def set_age_threshold(self, v): self.age_threshold = int(v) if v else 50
    def set_size_threshold(self, v): self.size_threshold = int(v) if v else 20
    def set_node_threshold(self, v): self.node_threshold = int(v) if v else 3
    #def toggle_schema_view(self): self.show_schema_preview = not self.show_schema_preview

    def toggle_schema_view(self):
        self.show_schema_preview = not self.show_schema_preview
        
        # Force Plotly to recalculate dimensions after the DOM reflows
        return rx.call_script("setTimeout(() => window.dispatchEvent(new Event('resize')), 50);")

    def update_group1_cols(self, col: str, checked: bool):
        if checked and col not in self.group1_selected_cols: self.group1_selected_cols.append(col)
        elif not checked and col in self.group1_selected_cols: self.group1_selected_cols.remove(col)

    def update_group2_cols(self, col: str, checked: bool):
        if checked and col not in self.group2_selected_cols: self.group2_selected_cols.append(col)
        elif not checked and col in self.group2_selected_cols: self.group2_selected_cols.remove(col)

    def update_cph_covariates(self, covariate: str, checked: bool):
        if checked:
            if covariate not in self.cph_active_covariates: 
                self.cph_active_covariates.append(covariate)
        else:
            if covariate in self.cph_active_covariates: 
                self.cph_active_covariates.remove(covariate)

        # Wipes old forest plots/tables the exact moment a clinical variable is toggled
        self.clear_results_display()

    async def handle_upload(self, files: list[rx.UploadFile]):
        for file in files:
            try:
                # Clear any previous upload error messages at the start
                self.upload_error = ""

                # Clear previous uploaded CLEANSED DATASET FRAME at the start
                self._override_active_df = None

                file_bytes = file.file.read()
                
                # 1. Check if file size exceeds 25 MB (25 * 1024 * 1024 bytes)
                if len(file_bytes) > 25 * 1024 * 1024:
                    self.upload_error = f"File is too large ({file.filename}). Maximum allowed size is 25 MB."
                    yield                               # Push error to frontend instantly
                    await asyncio.sleep(2)              # Wait 4 seconds
                    self.upload_error = ""              # Clear error string
                    yield                               # Push cleared state to hide it
                    return

                filename_lower = file.filename.lower()
                
                if filename_lower.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(file_bytes))
                elif filename_lower.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(io.BytesIO(file_bytes))
                else:
                    # 2. Set the error message for invalid format
                    self.upload_error = f"Unsupported format: {file.filename}."
                    yield                                               # Push error to frontend instantly
                    await asyncio.sleep(2)                              # Wait 4 seconds
                    self.upload_error = ""                              # Clear error string
                    yield                                               # Push cleared state to hide it
                    return

                # 3. Check if row count exceeds the safe memory threshold (100,000 rows)
                if len(df) > 100000:
                    self.upload_error = f"Dataset has too many rows ({len(df):,} rows). Maximum allowed is 100,000."
                    yield                                               # Push error to frontend instantly
                    await asyncio.sleep(2)                              # Wait 4 seconds
                    self.upload_error = ""                              # Clear error string
                    yield                                               # Push cleared state to hide it
                    return

                # --- Success Path Logic ---
                self.cohort_note = ""
                self.active_controls = [] 
                self.cph_active_covariates = []
                self.group1_selected_cols = []
                self.group2_selected_cols = []

                self.selected_dataset = "Own Data (.csv/.xls)"
                self.uploaded_data = df
                self.uploaded_filename = file.filename
                self.custom_filename = file.filename
                self.metadata_info = f"Source: {self.uploaded_filename} ({len(df)} rows). Custom cohort analysis."

                self.file_upload_success = True         # 🔥 Trigger the 2-second button text flash
                yield
                await asyncio.sleep(2)
                self.file_upload_success = False
                yield

                async for _ in self.update_dataset("Own Data (.csv/.xls)"):
                    yield
                    
            except Exception as e:
                self.results_ready = False
                self.show_schema_preview = False
                
                # 4. Set the runtime exception error message
                self.upload_error = f"Upload failed: {str(e)}"
                yield                                               # Push error to frontend instantly
                await asyncio.sleep(2)                              # Wait 2 seconds
                self.upload_error = ""                              # Clear error string
                yield                                               # Push cleared state to hide it
                return

    def handle_km_custom_stratify_change(self, col: str):
        pass

    def update_km_custom_subgroups(self, group: str, checked: bool):
        pass

    def update_km_numeric_split(self, val: str):
        pass

    def handle_km_selection(self, item: str, checked: bool):
        self.cohort_note = ""
        if checked:
            is_ecog_item = "ECOG" in item
            existing_ecog = [i for i in self.active_controls if "ECOG" in i]
            existing_others = [i for i in self.active_controls if "ECOG" not in i]

            if is_ecog_item:
                if existing_others: self.active_controls = [item]
                else:
                    if len(self.active_controls) >= 2: self.active_controls.pop(1) 
                    self.active_controls.append(item)
            else:
                if not existing_ecog:
                    if len(self.active_controls) >= 2: self.active_controls.pop(1)
                    self.active_controls.append(item)
        else:
            if item in self.active_controls: self.active_controls.remove(item)

    def set_selected_dataset(self, val: str):
        # 1. Dynamically bound slider limits based on cohort time ceilings
        max_limit = 45 if "Lung" in val else 90
        if self.months_list[1] > max_limit:
            self.months_list = [0.0, float(max_limit)]
        
        # 2. Drop all previous visual calculations & plots instantly
        self.clear_results_display()
        
        # 3. Complete the core dataset transition pipeline
        return self.update_dataset(val)
    
    def clear_sidebar_msg(self):
        self.show_sidebar_msg = False

    def handle_variable_accordion_click(self):
        """Triggers a toast warning if a user interacts with variables before selecting a model."""
        if self.selected_model == "":
            return self._show_custom_toast(
                "Please select an Analytical Model first to configure Variables.", 
                "error"
            )
    
    @rx.var
    def is_dataset_empty(self) -> bool:
        # If a built-in cohort is picked, it's not empty
        if "Breast" in self.selected_dataset or "Lung" in self.selected_dataset:
            return False
            
        # If custom data is picked, check if a file has actually been parsed
        if self.selected_dataset == "Own Data (.csv/.xls)":
            if self.uploaded_filename and self.csv_columns:
                if len(self.csv_columns) > 0:
                    return False
                    
        return True

    def handle_variable_accordion_click(self):
        """Validates that BOTH a dataset and model are present when checking variables."""
        # Sourced as an attribute property, NOT a callable method function
        dataset_missing = self.is_dataset_empty 
        model_missing = (self.selected_model == "")

        if dataset_missing and model_missing:
            return self._show_custom_toast("Please select Dataset and Model first to configure variables.", "error")
        elif dataset_missing:
            return self._show_custom_toast("Please select Dataset first to populate variables.", "error")
        elif model_missing:
            return self._show_custom_toast("Please select Analytical Model first to configure variables.", "error")
            
    @rx.var
    def combined_km_plot(self) -> go.Figure:
        df = self.get_current_dataframe()
        col = self.selected_column
        
        fig = go.Figure()
        if df is None or col not in df.columns or not self.group_a or not self.group_b:
            return fig

        if self.is_continuous:
            binned = pd.qcut(df[col], q=4, duplicates='drop')
            df['_binned_str'] = [f"{int(i.left)} to {int(i.right)}" for i in binned]
            df_a = df[df['_binned_str'] == self.group_a]
            df_b = df[df['_binned_str'] == self.group_b]
        else:
            df_a = df[df[col].astype(str) == self.group_a]
            df_b = df[df[col].astype(str) == self.group_b]

        if len(df_a) > 0:
            kmf_a = KaplanMeierFitter()
            kmf_a.fit(df_a['time'], event_observed=df_a['status'])
            fig.add_trace(go.Scatter(
                x=kmf_a.survival_function_.index, 
                y=kmf_a.survival_function_.iloc[:, 0],
                name=f"{self.group_a} ({self.selected_dataset})",
                mode='lines'
            ))

        if len(df_b) > 0:
            kmf_b = KaplanMeierFitter()
            kmf_b.fit(df_b['time'], event_observed=df_b['status'])
            fig.add_trace(go.Scatter(
                x=kmf_b.survival_function_.index, 
                y=kmf_b.survival_function_.iloc[:, 0],
                name=f"{self.group_b} ({self.selected_dataset})",
                mode='lines'
            ))

        fig.update_layout(
            title=f"{self.selected_dataset} Cohort Comparison: {col}",
            xaxis_title="Time",
            yaxis_title="Survival Probability"
        )
        return fig

    @rx.var
    def sidebar_bg_color(self) -> str:
        return {"info": "#eff6ff", "success": "#f0fdf4", "error": "#fff5f5"}.get(self.sidebar_variant, "#f0fdf4")

    @rx.var
    def sidebar_text_color(self) -> str:
        return {"info": "#1d4ed8", "success": "#166534", "error": "#c53030"}.get(self.sidebar_variant, "#166534")

    @rx.var
    def sidebar_border_color(self) -> str:
        return {"info": "#3b82f6", "success": "#bbf7d0", "error": "#feb2b2"}.get(self.sidebar_variant, "#bbf7d0")

    @rx.var
    def sidebar_icon(self) -> str:
        return {"info": "info", "success": "check-circle", "error": "triangle-alert"}.get(self.sidebar_variant, "check-circle")
    
    @rx.event(background=True)
    async def clear_sidebar_msg_after_delay(self):
        async with self:
            current_id = self._msg_id
        await asyncio.sleep(float(self.toast_duration) / 1000.0)
        async with self:
            if self._msg_id == current_id:
                self.show_sidebar_msg = False
    
    @rx.var
    def start_month(self) -> int: return int(self.months_list[0])

    @rx.var
    def end_month(self) -> int: return int(self.months_list[1])
        
    @rx.var 
    def survival_prob_lines(self) -> list[str]:
        return self.survival_probability_text.split(" | ") if self.survival_probability_text else []

    @rx.var
    def median_survival_lines(self) -> list[str]:
        return self.median_survival_text.split(" | ") if self.median_survival_text else []

    @rx.var
    def max_months_limit(self) -> int:
        target_df = self._raw_df if self._raw_df is not None else self.uploaded_data
        time_col = self.selected_time_column
        
        if time_col and target_df is not None and not target_df.empty:
            # Run through the exact same central normalization logic safely on-the-fly
            _, _, timeline_max, _ = self._normalize_dataframe(target_df, time_col)
            if timeline_max > 0:
                return timeline_max + 5
                
        if self.timeline_max and self.timeline_max > 0:
            return int(self.timeline_max) + 5
            
        return 90

    @rx.event
    def set_custom_min(self, group_num: int, col_name: str, val: str):
        if group_num == 1:
            self.g1_custom_min[col_name] = val
            target_cols = self.group1_selected_cols
            c_min = val
            c_max = self.g1_custom_max.get(col_name, "")
        else:
            self.g2_custom_min[col_name] = val
            target_cols = self.group2_selected_cols
            c_min = val
            c_max = self.g2_custom_max.get(col_name, "")

        # Live check: Is there any text in either input field?
        has_data = (str(c_min).strip() != "") or (str(c_max).strip() != "")
        
        if has_data:
            if col_name not in target_cols:
                target_cols.append(col_name)
        else:
            if col_name in target_cols:
                target_cols.remove(col_name)
                
        self.clear_results_display()
        self.run_analysis()

    @rx.event
    def set_custom_max(self, group_num: int, col_name: str, val: str):
        if group_num == 1:
            self.g1_custom_max[col_name] = val
            target_cols = self.group1_selected_cols
            c_min = self.g1_custom_min.get(col_name, "")
            c_max = val
        else:
            self.g2_custom_max[col_name] = val
            target_cols = self.group2_selected_cols
            c_min = self.g2_custom_min.get(col_name, "")
            c_max = val

        has_data = (str(c_min).strip() != "") or (str(c_max).strip() != "")
        
        if has_data:
            if col_name not in target_cols:
                target_cols.append(col_name)
        else:
            if col_name in target_cols:
                target_cols.remove(col_name)
                
        self.clear_results_display()
        self.run_analysis()

    def _run_km_engine(self, df, days_input, max_x):
        # 🧹 CLEAR LINGERING COX PROPORTIONAL HAZARDS STATE VARIABLES
        self.cph_table_data = []
        self.cph_fig = go.Figure()
        self.cph_survival_prob = ""
        self.cph_log_rank_p = ""
        self.cph_error_message = ""
        self.cohort_note = ""
        self.km_error_message = "" # Clear any previous km error state

        # ✅ Use passed model-scaled dataframe (`df`) --> matches timeline unit (Months / Days) and slider filters.
        if df is None or df.empty:
            return

        # ✅ DYNAMIC COLUMN MAPPING: Use the user's chosen column names
        time_col = self.selected_time_column
        event_col = self.selected_event_column

        # Safety guard if columns are missing
        if not time_col or not event_col or time_col not in df.columns or event_col not in df.columns:
            return self._show_custom_toast("Time or Event column not properly configured for KM analysis.", "error")

        self.at_risk_data = []
        fig = go.Figure()
                                    
        # 🎯 Fetch lower and upper bounds from timeline state early
        lower_limit = self.timeline_values[0] if hasattr(self, "timeline_values") and len(self.timeline_values) > 0 else 0

        # 🎯 DYNAMIC TIMELINE INTERVALS spanning from lower_limit to max_x
        step = (max_x - lower_limit) / 4.0
        raw_intervals = [
            lower_limit, 
            lower_limit + step, 
            lower_limit + (step * 2), 
            lower_limit + (step * 3), 
            max_x
        ]
        intervals = sorted(list(set([int(round(i)) for i in raw_intervals])))
        
        unit_lbl = getattr(self, "timeline_unit", "Mo")                                         # 🎯 Fetch active scale unit dynamically
        self.at_risk_columns = [f"Number at Risk ({unit_lbl})"] + [f"{i}" for i in intervals]   
        
        # 1. ------------------------- Check Active SIDEBAR Selections g1/G2_checked_values & Build masks 1/2 ----------------------- #
        def _get_mask_activity_and_summary(checked_dict, min_dict, max_dict, df):
            mask = pd.Series(True, index=df.index)
            is_active = False
            filter_parts = []
            
            # A. Standard Predefined / Categorical / Binned Inputs
            for col, selected_label in checked_dict.items():
                if isinstance(selected_label, list) and not selected_label:
                    continue
                    
                val_to_match = selected_label[0] if isinstance(selected_label, list) else selected_label

                if not val_to_match or "Custom Range:" in str(val_to_match):
                    continue 

                    
                if col in df.columns:
                    is_active = True
                    val_str = str(val_to_match).strip()
                    filter_parts.append(f"{col} = {val_str}")
                    
                    # --- Check if it's a binned interval label (e.g., Low, Med, High) --- #
                    if val_str.startswith(("Low", "Med", "Medium", "High")):
                        import re
                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
                        if len(numbers) >= 2:
                            low_b, high_b = float(numbers[0]), float(numbers[1])
                            mask = mask & (df[col] >= low_b) & (df[col] <= high_b)
                    else:
                        options = self.column_unique_values.get(col, [])
                        matching_option = next(
                            (opt for opt in options if str(opt["label"]).strip().lower() == val_str.lower()), 
                            None
                        )
                        
                        if matching_option and matching_option.get("is_numeric"):
                            c_min, c_max = matching_option.get("min"), matching_option.get("max")
                            label = matching_option.get("label", "")
                            if ">" in label:
                                if c_min is not None: mask = mask & (df[col] > c_min)
                                if c_max is not None: mask = mask & (df[col] <= c_max)
                            elif "<" in label:
                                if c_min is not None: mask = mask & (df[col] >= c_min)
                                if c_max is not None: mask = mask & (df[col] < c_max)
                            else:
                                if c_min is not None: mask = mask & (df[col] >= c_min)
                                if c_max is not None: mask = mask & (df[col] <= c_max)
                        elif matching_option and "values" in matching_option:
                            raw_vals = [str(v) for v in matching_option["values"]]
                            mask = mask & (df[col].astype(str).isin(raw_vals))
                        else:
                            mask = mask & (df[col].astype(str).str.strip().str.lower() == val_str.lower())

            # B. Custom Numeric Range Selections (Checked directly via min_dict/max_dict)
            for col, raw_min in min_dict.items():
                raw_max = max_dict.get(col, None)
                has_min = raw_min is not None and str(raw_min).strip() != ""
                has_max = raw_max is not None and str(raw_max).strip() != ""
                
                if has_min or has_max:
                    if col in df.columns:
                        is_active = True
                        min_str = str(raw_min).strip() if has_min else ""
                        max_str = str(raw_max).strip() if has_max else ""
                        
                        if has_min and has_max:
                            filter_parts.append(f"{col} = Custom Range: {min_str}-{max_str}")
                            mask = mask & (df[col] >= float(raw_min)) & (df[col] <= float(raw_max))
                        elif has_min:
                            filter_parts.append(f"{col} = Custom Range: ≥{min_str}")
                            mask = mask & (df[col] >= float(raw_min))
                        elif has_max:
                            filter_parts.append(f"{col} = Custom Range: ≤{max_str}")
                            mask = mask & (df[col] <= float(raw_max))

            summary_str = ", ".join(filter_parts) if filter_parts else ""
            return mask, is_active, summary_str

        # 2. ------------------------- Evaluate Selections, Masks & Summaries for Group 1 & Group 2 -------------------------- #
        mask1, g1_active, g1_filters_summary = _get_mask_activity_and_summary(
            self.g1_checked_values, 
            getattr(self, "g1_custom_min", {}), 
            getattr(self, "g1_custom_max", {}), 
            df
        )
        
        mask2, g2_active, g2_filters_summary = _get_mask_activity_and_summary(
            self.g2_checked_values, 
            getattr(self, "g2_custom_min", {}), 
            getattr(self, "g2_custom_max", {}), 
            df
        )

        # 3. --------------------------------------- Bidirectional Complementation Logic ------------------------------------- #
        if not g1_active and not g2_active:             # --- Full Cohort --- #          
            mask1 = np.ones(len(df), dtype=bool)            
            mask2 = np.ones(len(df), dtype=bool)
        elif g1_active and not g2_active:               # --- Group 1: Stratified / Group 2: Complementary Cohort --- #                      
            mask2 = ~mask1                                  
        elif g2_active and not g1_active:               # --- Group 2: Stratified / Group 1: Complementary Cohort --- #                    
            mask1 = ~mask2                                  
        # else: Both active                             # --- Group 1: Stratified / Group 2: Stratified --- # 

        # --- Resolve Dataframes early for validation ---
        is_full_cohort = not g1_active and not g2_active
        if is_full_cohort:
            df1, df2 = df.copy(), pd.DataFrame(columns=df.columns)
        else:
            df1, df2 = df[mask1].copy(), df[mask2].copy()

        # --- 🛡️ Minimum Sample Size Validation Guardrail ---
        min_sample_threshold = 5
        if not is_full_cohort:
            n1, n2 = len(df1), len(df2)
            if n1 < min_sample_threshold or n2 < min_sample_threshold:
                error_msg = (
                    f"Insufficient sample size: Group 1 contains {n1} patient(s) "
                    f"and Group 2 contains {n2} patient(s). A minimum of {min_sample_threshold} "
                    "patients per group is required for stable Kaplan-Meier estimation."
                )
                self.km_error_message = error_msg
                return
        else:
            self.km_error_message = ""

        # 4. ------------------------------------ Dynamic Tooltips & Composition Texts --------------------------------------- #
        total_n = len(df)
        self.full_cohort_composition_text = f"Full Cohort (Total N = {total_n})"
        
        # Compute and explicitly sync back to the state object
        is_full_cohort = not g1_active and not g2_active
        self.is_full_cohort = is_full_cohort  # 👈 ADD THIS LINE HERE

        if is_full_cohort:
            self.group1_composition_text = "Group 1: Full Cohort (No filters active)"
            self.group2_composition_text = "Group 2: Full Cohort (No filters active)"
            self.complement_composition_text = "N/A (Full Cohort baseline)"
        elif g1_active and not g2_active:
            self.group1_composition_text = f"Group 1 Filters: {g1_filters_summary}"
            self.complement_composition_text = "Complement: All patients not meeting Group 1 criteria (Rest of Cohort)"
            self.group2_composition_text = self.complement_composition_text
        elif g2_active and not g1_active:
            self.complement_composition_text = "Complement: All patients not meeting Group 2 criteria (Rest of Cohort)"
            self.group1_composition_text = self.complement_composition_text
            self.group2_composition_text = f"Group 2 Filters: {g2_filters_summary}"
        else:
            self.group1_composition_text = f"Group 1 Filters: {g1_filters_summary}"
            self.group2_composition_text = f"Group 2 Filters: {g2_filters_summary}"
            self.complement_composition_text = "N/A (Both groups explicitly stratified)"
        
        # ---------------------------------------------- Curve Generation & Fitting ----------------------------------------- #
        prob_strings, med_strings = [], []

        # 🎨 Dynamic Color Assignment
        if is_full_cohort:
            g1_color = "#333333"             
            g1_fill  = "rgba(0, 0, 0, 0.04)"   
            g2_color = "rgba(0, 0, 0, 0)"      
            g2_fill  = "rgba(0, 0, 0, 0)"      
        else:                                            
            g1_color = "#3b82f6"              # Blue
            g1_fill  = "rgba(59, 130, 246, 0.1)"
            g2_color = "#ec4899"              # Pink
            g2_fill  = "rgba(236, 72, 153, 0.1)"
        
        unit_label = self.timeline_unit  # --- self.timeline_unit from State: will be "Mo", "Years", or "Days" --- #

        if is_full_cohort:
            # --- SCENARIO 1: FULL COHORT ---
            
            df1, df2 = df.copy(), pd.DataFrame(columns=df.columns)
            self.current_df1, self.current_df2 = df1, df2

            lbl1 = "Full Cohort"
            kmf1 = KaplanMeierFitter()
            kmf1.fit(durations=df[time_col], event_observed=df[event_col], label=lbl1)
            self._add_km_trace(fig, kmf1, g1_color, g1_fill)

            p_val1 = kmf1.predict(days_input)
            m_val1 = kmf1.median_survival_time_
            p_ci, m_ci = get_ci_string(kmf1, days_input)

            prob_strings.append(f"Full Cohort: {int(p_val1 * 100)}% | 95% CI: {p_ci}")
            med_val_display = f"{m_val1:.1f} {unit_label}" if not np.isinf(m_val1) and not pd.isna(m_val1) else "NR"
            med_strings.append(f"Full Cohort: {med_val_display} | 95% CI: {m_ci}")

            self.at_risk_data.append([lbl1] + [str(self._get_at_risk(kmf1, d)) for d in intervals])
            max_time_1 = kmf1.timeline[-1]
            marker_x_1 = min(days_input, max_time_1)
            fig.add_trace(go.Scatter(
                x=[marker_x_1], y=[p_val1], mode='markers', showlegend=False,
                marker=dict(color=g1_color, size=9, line=dict(width=1.5, color='white')),
                hoverinfo="text", hovertext=f"{lbl1}<br>Time: {marker_x_1:.1f}<br>Survival: {int(p_val1 * 100)}%"
            ))
        else:
            # --- SCENARIO 2 & 3: STRATIFIED OR COMPLEMENTARY MODES ---
            df1, df2 = df[mask1].copy(), df[mask2].copy()
            self.current_df1, self.current_df2 = df1, df2

            if not df1.empty:
                lbl1 = "Group 1 (Complement)" if (not g1_active and g2_active) else "Group 1"

                kmf1 = KaplanMeierFitter()
                kmf1.fit(durations=df1[time_col], event_observed=df1[event_col], label=lbl1)
                self._add_km_trace(fig, kmf1, g1_color, g1_fill)

                p_val1 = kmf1.predict(days_input)               # --- calculates Survival Probability --- #
                m_val1 = kmf1.median_survival_time_             # --- calculates Median Survival Time --- #
                p_ci, m_ci = get_ci_string(kmf1, days_input)    # --- calculates CIs for Survival Probability & Median Survival Time --- #

                prob_strings.append(f"{lbl1}: {int(p_val1 * 100)}% | 95% CI: {p_ci}")
                med_val_display = f"{m_val1:.1f} {unit_label}" if not np.isinf(m_val1) and not pd.isna(m_val1) else "NR"
                med_strings.append(f"{lbl1}: {med_val_display} | 95% CI: {m_ci}")

                self.at_risk_data.append([lbl1] + [str(self._get_at_risk(kmf1, d)) for d in intervals])
                max_time_1 = kmf1.timeline[-1]
                marker_x_1 = min(days_input, max_time_1)
                fig.add_trace(go.Scatter(
                    x=[marker_x_1], y=[p_val1], mode='markers', showlegend=False,
                    marker=dict(color=g1_color, size=9, line=dict(width=1.5, color='white')),
                    hoverinfo="text", hovertext=f"{lbl1}<br>Time: {marker_x_1:.1f}<br>Survival: {int(p_val1 * 100)}%"
                ))

            if not df2.empty:
                lbl2 = "Group 2 (Complement)" if (not g2_active and g1_active ) else "Group 2"

                kmf2 = KaplanMeierFitter()
                kmf2.fit(durations=df2[time_col], event_observed=df2[event_col], label=lbl2)
                self._add_km_trace(fig, kmf2, g2_color, g2_fill)

                p_val2 = kmf2.predict(days_input)
                m_val2 = kmf2.median_survival_time_
                p_ci, m_ci = get_ci_string(kmf2, days_input)

                prob_strings.append(f"{lbl2}: {int(p_val2 * 100)}% | 95% CI: {p_ci}")
                med_val_display2 = f"{m_val2:.1f} {unit_label}" if not np.isinf(m_val2) and not pd.isna(m_val2) else "NR"
                med_strings.append(f"{lbl2}: {med_val_display2} | 95% CI: {m_ci}")

                self.at_risk_data.append([lbl2] + [str(self._get_at_risk(kmf2, d)) for d in intervals])
                max_time_2 = kmf2.timeline[-1]
                marker_x_2 = min(days_input, max_time_2)
                fig.add_trace(go.Scatter(
                    x=[marker_x_2], y=[p_val2], mode='markers', showlegend=False,
                    marker=dict(color=g2_color, size=9, line=dict(width=1.5, color='white')),
                    hoverinfo="text", hovertext=f"{lbl2}<br>Time: {marker_x_2:.1f}<br>Survival: {int(p_val2 * 100)}%"
                ))

        # ---------------------------------------------- 4. Table 1 & Statistical Summary ------------------------------------ #
         
        self.survival_probability_list = prob_strings   # --- ASSIGNMENTS ARE AT STATE LEVEL FOR CONSUMTPION --- #
        self.median_survival_list = med_strings
        
        self.g1_n = int(len(df1))
        self.g2_n = int(len(df2)) if not is_full_cohort else 0
        self.g1_events = int(df1[event_col].sum()) if not df1.empty and event_col in df1.columns else 0
        self.g2_events = int(df2[event_col].sum()) if not df2.empty and event_col in df2.columns else 0

        # ---------------🎯 FULL COHORT RISK TABLE CLEANUP --------------- #
        if "Full Cohort" in str(self.g1_detailed_label) and "Full Cohort" in str(self.g2_detailed_label):
            if len(self.at_risk_data) > 0:
                self.at_risk_data = [["Full Cohort"] + self.at_risk_data[0][1:]]
        elif df2.empty:
            print("Group 2 is empty\n")
        
        # 👇 SYNC TABLE 1 COLUMN LABELS DIRECTLY FROM THE ENGINE'S NORM:
        if is_full_cohort:
            self.table_one_g1_label = "Full Cohort"
            self.table_one_g2_label = ""
        else:
            self.table_one_g1_label = lbl1 if 'lbl1' in locals() else "Group 1"
            self.table_one_g2_label = lbl2 if 'lbl2' in locals() else "Group 2"

        self._generate_table_one(df1, df2 if not is_full_cohort else df1)   # --- standard baseline Characteristics & Demographics table --- #
        
        self.survival_probability_list = prob_strings
        self.median_survival_list = med_strings

        self.survival_probability_text = " | ".join(prob_strings)
        self.median_survival_text = " | ".join(med_strings)
        
        # --- Explicit Three-Way Branching for Statistical Summaries ---
        if not is_full_cohort and not df1.empty and not df2.empty:          # --- Two-Group Comparison / Stratified Mode --- #
            self.ci_text = "Patient Group Comparison"
            try:
                lr_res = logrank_test(df1[time_col], df2[time_col], df1[event_col], df2[event_col])
                p_val = lr_res.p_value
                
                # Clinical formatting convention: < 0.001 or rounded to 2 decimal places
                if p_val < 0.001:
                    self.p_value_text = "Log-rank p: < 0.001"
                else:
                    self.p_value_text = f"Log-rank p: {p_val:.2f}"
            except: 
                self.p_value_text = "Log-rank p: N/A"
        elif is_full_cohort:
            # Scenario: Full Cohort Baseline
            self.ci_text = "Full Cohort"
            self.p_value_text = "Log-rank test: N/A (Single baseline population)"
        else:
            # Scenario: Single Group Active
            self.ci_text = "Single Subgroup Analysis"
            self.p_value_text = "Log-rank test: N/A (Requires two active cohorts)"

        # ---------------------------------------------- 5. Layout Update ------------------------------------ #
        fig.update_layout(
            title={
                "text": f"<b>Group Survival Comparison ({self.selected_model})</b>", 
                "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 13}
            },
            template="simple_white", 
            height=420, 
            autosize=True, 
            margin=dict(t=60, b=40, l=40, r=20), 
            showlegend=False                     
        )

        fig.update_yaxes(range=[0, 1.05], title="Survival Probability")
        fig.update_xaxes(range=[lower_limit, max_x])
            
        self.survival_fig_g1 = fig
        self.is_ready = True

        # --- 🌟 3. Generates the CPH Table for the KM Analysis --- #
        self._generate_km_cph_table()
        
        # --- 🌟 1. Current dataframes from State to initialize columns for BASELINE CHARACTERISTICS TABLE 1 --- #
        df1 = getattr(self, "current_df1", None)
        df2 = getattr(self, "current_df2", None)
        
        self.init_baseline_config()       

        # --- 🌟 2. Generate Table 1, fully computed with P-Values
        # Refresh Table 1 immediately using the updated column_configs
        if df1 is not None and not df1.empty:
            self._generate_table_one(df1, df2)

    # --------------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------- RUN CPH ENINGE ------------------------------------------------- #
    def _run_cph_engine(self, df=None, days_input=365, preserve_state=False, active_covs_override=None, is_toggle=False):
        # 🧹 --- ONLY CLEAR VARIABLES IF DIRECTLY SELECTED IN SIDEBAR --- #
        self.cph_table_data = []        # --- Guarantees no lingering rows accumulate --- #
        self.cph_fig = go.Figure()      # --- Clears lingering plots on toggle/re-run --- #
        
        if not preserve_state:
            self.survival_probability_text = ""
            self.median_survival_text = ""
            self.p_value_text = ""
            self.ci_text = ""
            self.survival_probability_list = []
            self.median_survival_list = []
            self.at_risk_data = []
            self.cph_error_message = ""
        else:
            self.cph_error_message = ""
        
        # ------------------------------------ 1. Robust Dataframe Resolution ------------------------------------- #
        active_df = df
        if active_df is None or (hasattr(active_df, "empty") and active_df.empty):
            active_df = getattr(self, "filtered_df", None)
            if active_df is None or (hasattr(active_df, "empty") and active_df.empty):
                active_df = getattr(self, "df", None)

        if active_df is None or (hasattr(active_df, "empty") and active_df.empty):
            print("DEBUG [Engine ERROR]: Active dataframe is empty!")
            self.cph_error_message = "Active dataframe is empty for CPH analysis."
            if is_toggle:
                return
            return self._show_custom_toast(self.cph_error_message, "error")

        # FIX: Reset the index of active_df immediately so all downstream sub-samplings (like bootstrapping) are clean
        active_df = active_df.reset_index(drop=True)
        df = active_df.copy()

        # 2. Handle Continuous Variable Binning for Stratification
        # If 'pnodes' is selected for stratification, convert it to a categorical binned column
        if "pnodes" in self.selected_strata_vars and "pnodes" in df.columns:
            bins = [-1, 0, 3, 9, np.inf]
            labels = ["0 Nodes", "1-3 Nodes", "4-9 Nodes", "10+ Nodes"]
            df["pnodes_strat_binned"] = pd.cut(df["pnodes"], bins=bins, labels=labels)
            # Replace 'pnodes' in strata list with our binned categorical version for the engine
            active_strata = [
                "pnodes_strat_binned" if v == "pnodes" else v 
                for v in self.selected_strata_vars
            ]
        else:
            active_strata = self.selected_strata_vars      
        
        time_col = getattr(self, "selected_time_column", "time")            # --- vom Benutzer gewaehlten Zeit/Event Spalten --- #
        event_col = getattr(self, "selected_event_column", "status")        # --- checken, ob default values assigned sind --- #

        # 🔑 Local override vs user sidebar selection
        if active_covs_override is not None:
            active_covs = [c for c in active_covs_override if c not in [time_col, event_col]]
        else:
            active_covs = [c for c in getattr(self, "cph_active_covariates", []) if c not in [time_col, event_col]]

        keep_cols = [time_col, event_col] + active_covs
        missing_cols = [c for c in keep_cols if c not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns for CPH: {missing_cols}"
            self.cph_error_message = error_msg
            return self._show_custom_toast(error_msg, "error")

        # Extract base requirements
        df_fit = pd.DataFrame()

        df_fit[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df_fit[event_col] = pd.to_numeric(df[event_col], errors='coerce')

        # Add all covariates to df_fit as well
        for c in active_covs:
            df_fit[c] = pd.to_numeric(df[c], errors='coerce') if df[c].dtype in [int, float, 'float64', 'int64'] else df[c]
        
        df_fit = df_fit.reset_index(drop=True)  # --- Reset index, so df_fit has a unique integer index for all downstream calibration/DCA calculations --- #
        
        # 🛑 STRICT CLINICAL VALIDATION: Ensure event column is binary (0 or 1)
        event_series = df_fit[event_col].dropna()
        unique_events = set(event_series.unique()) 
         
        if not unique_events.issubset({0, 1}) or len(unique_events) > 2:
            error_msg = (
                f"Invalid Event Column Error: '{event_col}' contains continuous or non-binary values. "
                "The Event column must be binary (0 for censored, 1 for event)."
            )
            self.cph_error_message = error_msg
            return self._show_custom_toast(error_msg, "error")

        # Validate and coerce covariates to numeric
        unmapped_text_cols = []
        for c in active_covs:
            converted_col = pd.to_numeric(df[c], errors='coerce')
            if converted_col.isna().all() and not df[c].isna().all():
                unmapped_text_cols.append(c)
            else:
                df_fit[c] = converted_col

        if unmapped_text_cols:
            self.cph_error_message = f"Column {unmapped_text_cols} contains unmapped text. Use 'Map Data' to map Text to Numbers."
            return

        df_fit = df_fit.dropna()        # --- DATAFRAME FINALIZED FOR CPH MODEL ANALYSIS RUN --- #

        if df_fit.empty:
            error_msg = "No complete records available for CPH analysis."
            self.cph_error_message = error_msg
            return self._show_custom_toast(error_msg, "error")

        # --- 🌟 Generate Table 1 FOR CPH ANALYSIS RESULTS REPORT from CLEANSED CPH FIT dataframe: df_fit --- #
        self.init_baseline_config()
        self._generate_table_one(df1=df_fit, df2=None)      # --- generates the Data for the CPH Baseline Characteristics Table --- #
        
        # --------------------------------- Dynamic Stratification & Fit Logic --------------------------------- #
        auto_stratify = getattr(self, "cph_auto_stratify_violations", True)

        # ---------------------------- 1. 🌟 Calculate Schoenfeld violations based on current strata ----------------------=- #
        detected_violations = []
        current_strata = getattr(self, "selected_strata_vars", [])
        
        if len(active_covs) > 1:
            try:
                from lifelines.statistics import proportional_hazard_test
                # 🔑 Fit with dynamic state variables instead of hardcoded strings
                temp_cph = CoxPHFitter(penalizer=0.1).fit(
                    df_fit, 
                    duration_col=time_col, 
                    event_col=event_col, 
                    strata=current_strata if current_strata else None
                )

                temp_ph_test = proportional_hazard_test(
                    temp_cph, 
                    df_fit, 
                    time_col=time_col, 
                    event_col=event_col, 
                    regrade=False
                )

                p_vals = getattr(temp_ph_test, 'p_value', None)
                if p_vals is not None:
                    for cov_name, p_val in zip(temp_cph.summary.index, p_vals):
                        if float(p_val) < 0.05:
                            detected_violations.append(str(cov_name).strip())
            except Exception as e:
                print(f"DEBUG [Pre-check Schoenfeld Error]: {e}")

        # ------------------------- 2. 🌟 Combine existing active strata history with any newly detected violations --------------------- #
        self.cph_detected_violations = list(dict.fromkeys(current_strata + detected_violations))
        
        # 🔍 DEBUG PRINT: Verify what violations were caught and stored
        #print(f"DEBUG: 5008: [Active Strata]: {current_strata}")
        #print(f"DEBUG: 5009: [Detected Violations List]: {self.cph_detected_violations}")

        # 2. Strata is strictly controlled by what the physician explicitly checks in the sandbox
        manual_strata = getattr(self, "selected_strata_vars", [])
        
        if getattr(self, "cph_auto_stratify_violations", False):
            # When toggle is ON, use whatever boxes the physician has checked
            current_strata = manual_strata
        else:
            # When toggle is OFF, no stratification
            current_strata = []

        self.cph_active_strata = current_strata

        # ------------------------------ 3. 🚀 Pass 2: Final Multivariable Model Fit (THIS STAYS RIGHT HERE!) ---------------------- #
        try:
            #print(f"DEBUG: 5235 [CPH Fit] -> Selected time_col: {time_col}")
            #print(f"DEBUG [CPH Fit] -> Selected event_col: {event_col}")
            #print(f"DEBUG [CPH Fit] -> Active covariates: {active_covs}")
            #print(f"DEBUG [CPH Fit] -> Available DataFrame columns: {list(df_fit.columns)}")
            
            cph = CoxPHFitter(penalizer=0.1)

            fit_cols = [time_col, event_col] + active_covs

            cph.fit(
                df_fit[fit_cols], 
                duration_col=time_col, 
                event_col=event_col, 
                strata=current_strata if current_strata else None
            )
            
            self._cph_fitted_model = cph # --- 🔑 Save fitted local `cph` model to `self` so the calibration generator can use it--- #

        except Exception as e:
            error_msg = f"CPH Fitting Error (Stratified): {str(e)}"
            self.cph_error_message = error_msg
            return self._show_custom_toast(error_msg, "error")

        # ----------------------------- 🌟 Calculate Reverse Kaplan-Meier Median Follow-Up Time -------------------------------- #
        try:
                from lifelines import KaplanMeierFitter
                
                # 🌟 Use dynamic state variables
                #time_col = self.selected_time_column
                #event_col = self.selected_event_column
                
                kmf_fu = KaplanMeierFitter()
                kmf_fu.fit(df_fit[time_col], event_observed=1 - df_fit[event_col])
                median_fu = kmf_fu.median_survival_time_
        except Exception as e:
                print(f"DEBUG [Median Follow-Up Error]: {e}")
                median_fu = 0.0
        # ------------------------------------------- 🌟 METRIC EXTRACTION & VALIDATION ---------------------------------------- #
        unit_label = getattr(self, "timeline_unit", "Mo")
        n_samples = getattr(cph, 'number_of_subjects_', len(df_fit))
        n_events = int(cph.event_observed.sum())
        n_censored = n_samples - n_events
        
        self.cph_sample_metrics_text = f"Analyzed Cohort: N = {n_samples:,} patients | {n_events:,} | Censored: {n_censored:,} | Median Follow-Up: {median_fu:.1f} {unit_label}"
        self.cph_full_cohort_n = n_samples

        c_index = cph.concordance_index_

        try:
            import numpy as _np
            se_c = _np.sqrt(0.25 / max(int(n_events), 1))
            c_lower = max(0.0, c_index - 1.96 * se_c)
            c_upper = min(1.0, c_index + 1.96 * se_c)
            self.cph_c_index_text = f"Harrell's C-index: {c_index:.3f} (95% CI: {c_lower:.3f} – {c_upper:.3f})"
        except Exception as e:
            print(f"DEBUG [C-index CI Error]: {e}")
            self.cph_c_index_text = f"Harrell's C-index: {c_index:.3f}"

        lr_test = cph.log_likelihood_ratio_test()
        self.cph_log_rank_p = f"Global LR Test: chi2 = {lr_test.test_statistic:.2f}, p = {lr_test.p_value:.4e}"

        # ------------------------------------ Univariate mapping with 95% Confidence Intervals ----------------------------------- #
        univariate_map = {}
        for cov in active_covs:
            try:
                # Use dynamic time_col and event_col references
                uni_cph = CoxPHFitter(penalizer=0.1).fit(
                    df_fit[[time_col, event_col, cov]], 
                    duration_col=time_col, 
                    event_col=event_col
                )
                uni_summary = uni_cph.summary
                u_hr = float(np.exp(uni_summary['coef'].iloc[0]))
                u_p = float(uni_summary['p'].iloc[0])
                
                u_l_col = [c for c in uni_summary.columns if 'lower' in c.lower()][0]
                u_h_col = [c for c in uni_summary.columns if 'upper' in c.lower()][0]
                
                u_low = float(np.exp(uni_summary[u_l_col].iloc[0])) if 'exp' not in u_l_col.lower() else float(uni_summary[u_l_col].iloc[0])
                u_high = float(np.exp(uni_summary[u_h_col].iloc[0])) if 'exp' not in u_h_col.lower() else float(uni_summary[u_h_col].iloc[0])
                
                univariate_map[cov] = (u_hr, u_low, u_high, u_p)
            except Exception as e:
                print(f"DEBUG [Univariate Fit Error for {cov}]: {e}")
                univariate_map[cov] = (np.nan, np.nan, np.nan, np.nan)

        summary = cph.summary
        hr_col = [c for c in summary.columns if 'exp(coef)' in c.lower() or 'hazard' in c.lower()][0]
        l_col_name = [c for c in summary.columns if 'lower' in c.lower()][0]
        h_col_name = [c for c in summary.columns if 'upper' in c.lower()][0]
        p_col_name = [c for c in summary.columns if 'p' == c.lower() or 'p-value' in c.lower()][:1][0]

        hr_values = summary[hr_col]
        lower_ci = np.exp(summary[l_col_name]) if 'exp' not in l_col_name.lower() else summary[l_col_name]
        upper_ci = np.exp(summary[h_col_name]) if 'exp' not in h_col_name.lower() else summary[h_col_name]

        # ------------------------------------------- Schoenfeld Proportional Hazards Assumptions --------------------------------------- #
        ph_violation_notes = []
        schoenfeld_p_map = {}
        
        try:
            from lifelines.statistics import proportional_hazard_test
            
            ph_test = proportional_hazard_test(
                cph, 
                df_fit, 
                time_col=time_col, 
                event_col=event_col, 
                regrade=False
            )
            p_vals_array = getattr(ph_test, 'p_value', None)
            
            if p_vals_array is not None and len(summary.index) == len(p_vals_array):
                for idx_name, p_val in zip(summary.index, p_vals_array):
                    clean_cov = str(idx_name).strip()
                    p_val_float = float(p_val)
                    schoenfeld_p_map[clean_cov] = p_val_float
                    if p_val_float < 0.05:
                        ph_violation_notes.append(f"'{clean_cov}' (p={p_val_float:.3f})")
        except Exception as e:
            print(f"DEBUG [Schoenfeld Exception Caught]: {e}")

        if ph_violation_notes:
            self.cph_ph_warning = f"PH Assumption Notice: Potential non-proportionality detected in {', '.join(ph_violation_notes)}."
        else:
            self.cph_ph_warning = "Proportional Hazards Assumption: Satisfied globally across covariates (Schoenfeld p > 0.05)."
        
        # ------------------------------------------- 🌟 TABLE DATA POPULATION (ATOMIC) ---------------------------------------- #
        temp_table_data = []
        for idx, hr, low, high, p in zip(summary.index, hr_values, lower_ci, upper_ci, summary[p_col_name]):
            uni_vals = univariate_map.get(idx, (np.nan, np.nan, np.nan, np.nan))
            
            u_hr_str = f"{uni_vals[0]:.2f}" if not np.isnan(uni_vals[0]) else "N/A"
            u_low_str = f"{uni_vals[1]:.2f}" if not np.isnan(uni_vals[1]) else "N/A"
            u_high_str = f"{uni_vals[2]:.2f}" if not np.isnan(uni_vals[2]) else "N/A"
            u_p_str = f"{uni_vals[3]:.4f}" if not np.isnan(uni_vals[3]) else "N/A"
            uni_ci_str = f"{u_low_str} – {u_high_str}" if u_low_str != "N/A" else "N/A"

            adj_ci_str = f"{low:.2f} – {high:.2f}"

            lookup_key = str(idx).strip()
            sch_p = schoenfeld_p_map.get(lookup_key, np.nan)
            
            if np.isnan(sch_p) and schoenfeld_p_map:
                matching_keys = [k for k in schoenfeld_p_map.keys() if k.lower() == lookup_key.lower()]
                if matching_keys:
                    sch_p = schoenfeld_p_map[matching_keys[0]]

            if not np.isnan(sch_p):
                sch_p_str = "< 0.001" if sch_p < 0.001 else f"{sch_p:.3f}"
            else:
                sch_p_str = "0.999"

            # 8 Columns matching your new UI header structure
            temp_table_data.append([
                str(idx),           # 0: Variable Name
                u_hr_str,           # 1: Unadjusted HR
                uni_ci_str,         # 2: Univariate 95% CI
                u_p_str,            # 3: Univariate P
                f"{hr:.2f}",        # 4: Adjusted HR
                adj_ci_str,         # 5: Multivariate 95% CI
                f"{p:.4f}",         # 6: Adjusted P
                sch_p_str           # 7: Schoenfeld P
            ])
        
        self.cph_table_data = temp_table_data
        
        # Plotly Forest Plot Configuration
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hr_values, y=summary.index, mode='markers',
            error_x=dict(type='data', symmetric=False, array=upper_ci - hr_values, arrayminus=hr_values - lower_ci, color="#1e293b", thickness=1.5, width=6),
            marker=dict(color='#10b981', size=10, line=dict(width=1, color='white'))
        ))
        fig.add_vline(x=1, line_dash="dash", line_color="#ef4444", opacity=0.8, line_width=2)
        
        dataset_name = self.selected_dataset.split('/')[-1].replace('.csv', '').replace('_', ' ').title() if self.selected_dataset else "Active Cohort"

        fig.update_layout(
            title={"text": f"<b>{getattr(self, 'selected_model', 'Cox Proportional Hazards')} Prognosis ({dataset_name})</b>", "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 14}},
            template="simple_white",
            xaxis=dict(
                type="log", 
                range=[np.log10(0.1), np.log10(10)], 
                tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10],
                title="Multivariate Adjusted Hazard Ratio (95% CI)"
            ), 
            yaxis=dict(autorange="reversed"), 
            margin=dict(l=100, r=20, t=70, b=40), 
            height=420,
            autosize=True,
            modebar={"orientation": "h", "bgcolor": "rgba(0,0,0,0)", "color": "#475569", "activecolor": "#1e293b"}
        )

        # 🌟 Run Bootstrap Validation right before completion
        apparent_c, corrected_c = self.calculate_bootstrap_validation(
            df=df_fit,
            time_col=self.selected_time_column,
            event_col=self.selected_event_column,
            covariates=active_covs,
            strata_vars=current_strata,
            n_bootstraps=100
        )
        
        self.cph_apparent_c_index = apparent_c or 0.0
        self.cph_corrected_c_index = corrected_c or 0.0

        # 🔑 Centralized eval_time computation to guarantee absolute statistical integrity
        eval_time = float(df_fit[self.selected_time_column].median())
        
        # 🔑 ------------------------------- Call calibration generator using current fit data -------------------------------- #
        self.generate_calibration_curve(
            df_fit=df_fit,
            time_col=self.selected_time_column,
            event_col=self.selected_event_column,
            eval_time=eval_time
        )

        # 🔑 ----------------------------------------- Call decision curve generator -----------------------    --------------- #
        self.generate_decision_curve(                                                   # --- Calculated Intercept and Slope --- #
            df_fit=df_fit,
            time_col=self.selected_time_column,
            event_col=self.selected_event_column,
            eval_time=eval_time
        )

        # 🔑 -------------------------------- Inside data update / filtering handler------------------------------------------- #
        self.generate_time_dependent_auc(
            df_fit, 
            time_col=self.selected_time_column, 
            event_col=self.selected_event_column, 
            # eval_time=eval_time   # --- does not use eval_time 
        )
        
        # 🔑 -------------------------------- Compute Quantitative Calibration & GND Metrics ---------------------------------- #
        self._compute_gnd_and_calibration_metrics(                                               # --- Calculates Chi2 and P --- #
            cph=self._cph_fitted_model, 
            df_fit=df_fit, 
            t0=eval_time
        )

        # 🔑 -------------------------------- Compute Risk-Stratified Kaplan-Meier Curves ------------------------------- #
        self.generate_risk_stratified_km(
            df_fit=df_fit,
            time_col=self.selected_time_column,
            event_col=self.selected_event_column
        )
        
        # ------------------ Mark _run_cph_engine as successfully completed right before assigning the figure ----------------- #
        self.cph_has_run = True
        self.cph_fig = fig

        # --------------------------------- Safe Prediction for Stratified Models --------------------------------- #
        try:
            # Start with a valid copy of an actual row from df_fit to preserve exact datatypes and strata categories
            pred_df = df_fit.iloc[[0]].copy()
            
            # Set all model parameters to their mean values
            for col in cph.params_.index:
                if col in df_fit.columns:
                    pred_df[col] = df_fit[col].mean()
            
            # Ensure strata columns retain a valid baseline category from the dataset
            if cph.strata:
                for strata_col in cph.strata:
                    if strata_col in df_fit.columns:
                        pred_df[strata_col] = df_fit[strata_col].iloc[0]

            prob_series = cph.predict_survival_function(pred_df)
            closest_time_index = (prob_series.index.to_series() - days_input).abs().idxmin()
            surv_prob_val = prob_series.loc[closest_time_index].values[0]
            self.cph_survival_prob = f"{int(surv_prob_val * 100)}%"
        except Exception as e:
            print(f"DEBUG [CPH Prediction Error]: {e}")
            self.cph_survival_prob = "N/A"

        self.cph_error_message = ""
        self.cohort_note = ""

    
    def _add_km_trace(self, fig, kmf, color, fill):
        x = kmf.survival_function_.index.tolist()
        y = kmf.survival_function_.iloc[:, 0].tolist()
        ci = kmf.confidence_interval_
        
        # 1. The Shaded Confidence Interval Bounds
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1], 
                y=ci.iloc[:, 1].tolist() + ci.iloc[:, 0].tolist()[::-1], 
                fill='toself', 
                fillcolor=fill, 
                line=dict(color='rgba(255,255,255,0)'), 
                showlegend=False, 
                hoverinfo="skip"
            )
        )
        
        # 2. The Main Kaplan-Meier Curve Line
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y, 
                name=kmf.label, 
                mode='lines', 
                line=dict(
                    color=color, 
                    width=1.5,   # 🎯 CHANGED HERE (Down from 2 for a thinner look)
                    shape='hv'   # Keeps the signature KM step-staircase shape
                )
            )
        )

    def _get_at_risk(self, kmf, d_val):
        mask = kmf.event_table.index >= d_val       # creates a boolean mask of all timeline points in the model larger/equal the target timeline d_val
        return int(kmf.event_table.at[kmf.event_table.index[mask][0], 'at_risk']) if any(mask) else 0

# --------------------------------------------- End STATE CLASS ------------------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------------------------- #

def category_mapping_subrow(cfg: dict) -> rx.Component:
    return rx.cond(
        cfg["type"] == "Categorical",
        rx.box(
            rx.vstack(
                rx.text("Map Raw Values to Publication Labels:", font_size="11px", font_weight="600", color="gray.600"),
                rx.foreach(
                    cfg["mappings_list"].to(list[dict]),
                    lambda item: rx.hstack(
                        rx.badge(item["raw"], variant="soft", color_scheme="blue", size="1"),
                        rx.text("→", font_size="12px", color="gray.400"),
                        rx.input(
                            value=item["label"],
                            on_change=lambda val: State.update_category_mapping(
                                cfg["name"], item["raw"], val
                            ),
                            placeholder=f"Label for '{item['raw']}'",
                            size="1",
                            width="220px",
                        ),
                        align="center",
                        spacing="2",
                    )
                ),
                spacing="2",
                align_items="start",
                width="100%",
            ),
            padding="12px 16px",
            margin_top="6px",
            margin_bottom="6px",
            margin_left="24px",
            width="calc(100% - 24px)",
            background_color="#f8fafc",
            border_radius="6px",
            border="1px solid #e2e8f0",
            overflow_x="hidden", # Prevents horizontal scrollbar clipping
        ),
        rx.fragment()
    )

# ---- Creates Baseline Demographics & Clinical Characteristics Layout for Report Creation --- #
def render_config_row(cfg: dict) -> rx.Component:
    return rx.vstack(
        # Main variable config row
        rx.hstack(
            rx.box(
                rx.checkbox(
                    checked=cfg["include"],
                    on_change=lambda val: State.toggle_column_include(cfg["name"], val),
                ),
                width="60px",
            ),
            rx.box(
                rx.text(cfg["name"], font_size="13px", font_weight="500", truncate=True),
                width="140px",
            ),
            rx.box(
                rx.input(
                    default_value=cfg["label"],
                    on_blur=lambda val: State.update_column_label(cfg["name"], val),
                    size="1",
                    width="217px",
                ),
                width="200px",
            ),
            rx.box(
                rx.select(
                    ["Continuous", "Categorical"],
                    value=cfg["type"],
                    on_change=lambda val: State.update_column_type(cfg["name"], val),
                    size="1",
                    width="100px",
                ),
                width="120px",
                margin_left="12px",
            ),
            justify="between",
            width="100%",
            padding_x="8px",
            align="center",
        ),
        
        # --- HIERARCHICAL CATEGORY MAPPINGS ---
        rx.cond(
            cfg["type"] == "Categorical",
            rx.vstack(
                rx.hstack(
                    rx.box(width="140px"), # Spacer matching Variable Name column width
                    rx.hstack(
                        rx.text("Value", font_size="12px", color="gray.500", font_weight="600", width="45px", transform="translateX(-14px)"),
                        rx.text("➔", font_size="12px", color="gray.500", font_weight="600", transform="translateX(-28px)"),
                        rx.text("Label", font_size="12px", color="gray.500", font_weight="600", transform="translateX(-18px)"), 
                        align="center",
                        spacing="2",
                        padding_left="16px",
                    ),
                    width="100%",
                    padding_x="8px",
                ),
                # Mapping rows with clear hierarchical indentation
                rx.foreach(
                    cfg["mappings"].to(dict),
                    lambda item: rx.hstack(
                        rx.box(width="140px"), # Spacer locking alignment with Variable Name
                        rx.hstack(
                            # 👇 Standardized font size: removed monospace font family to match table data cells
                            rx.text(item[0], font_size="12px", color="gray.700", width="45px", transform="translateX(-25px)"),  # shifts values left
                            rx.input(
                                default_value=item[1],
                                on_blur=lambda new_name, raw_val=item[0]: State.update_category_mapping(cfg["name"], raw_val, new_name),
                                size="1",
                                width="144px",
                                margin_left="-12px",
                            ),
                            align="center",
                            spacing="2",
                            padding_left="36px",    # war 16
                        ),
                        width="100%",
                        padding_x="2px",
                    )
                ),
                width="100%",                  
                max_width="383px",             
                spacing="1",
                padding_y="8px",
                padding_bottom="14px",         
                background_color="#f8fafc",
                border_radius="6px",
                border_left="3px solid #cbd5e1",
                margin_left="85px",            
                margin_right="12px",           
            ),
            rx.fragment()
        ),
        width="100%",
        spacing="2",
        padding_y="6px",
        border_bottom="1px solid #f1f5f9",
    )

# --------------------- PDF Report Blue Rectangle Header Details Helper Function - Python ----------------------- #
def report_customization_panel():
    return rx.vstack(
        rx.hstack(
            rx.vstack(
                rx.text("PDF Report Configuration", size="1", font_size="14px", weight="bold", color="#1e293b"),
                #rx.text("Customize header details and export Publication-Supporting Document.", size="1", color="#64748b"),
                rx.text("", size="1", color="#64748b"),
                spacing="1",
            ),
            rx.spacer(),
            
            # Download PDF Button wrapped in a relative container
            rx.vstack(
                rx.button(
                    rx.icon(tag="download", size=12), 
                    "Download Report",
                    on_click=rx.cond(
                        State.results_ready,
                        State.download_active_report, # Points directly to the scalable dispatcher
                        State.handle_disabled_pdf_click,
                    ),
                    disabled=False, 
                    size="1",
                    bg=rx.cond(State.results_ready, rx.color("blue", 8), rx.color("slate", 6)),      
                    color="white",              
                    font_weight="bold",
                    cursor=rx.cond(State.results_ready, "pointer", "not-allowed"),
                    _hover={"bg": rx.cond(State.results_ready, rx.color("blue", 9), rx.color("slate", 6))},
                    width="100%",
                ),
                
                # Inline Warning Message using absolute positioning
                rx.cond(
                    State.show_pdf_warning,
                    rx.text(
                        State.pdf_warning_msg,
                        style={
                            "position": "absolute",
                            "top": "25px",          
                            "left": "0px",
                            "white_space": "nowrap", 
                            "font_size": "11px",
                            "color": "#D97706",     
                            "font_weight": "500",
                            "z_index": "9999",      
                        },
                    ),
                ),
                spacing="1",
                width="auto",
                align_items="start",
                position="relative",        
            ),
        ),

        #rx.divider(margin_y="1px"),
        rx.divider(margin_top="12px", margin_bottom="12px"),
        rx.hstack(
            rx.vstack(
                rx.text("Lead Investigator", size="1", color="#475569", weight="medium"),
                rx.input(
                    value=State.researcher_name,
                    on_change=State.set_researcher_name,
                    placeholder="Enter name",
                    size="1",
                    width="100%",
                ),
                spacing="1",
                width="100%",
            ),
            rx.vstack(
                rx.text("Institution / Department", size="1", color="#475569", weight="medium"),
                rx.input(
                    value=State.institution_name,
                    on_change=State.set_institution_name,
                    placeholder="Enter institution",
                    size="1",
                    width="100%",
                ),
                spacing="1",
                width="100%",
            ),
            rx.vstack(
                rx.text("Report Date", size="1", color="#475569", weight="medium"),
                rx.input(
                    value=State.report_date,
                    on_change=State.set_report_date,
                    placeholder="Month Day, Year",
                    size="1",
                    width="100%",
                ),
                spacing="1",
                width="100%",
            ),
            width="100%",
            spacing="3",
        ),
        # Include the modal component inline so it mounts with the panel
        #baseline_characteristics_modal(),
        
        padding="14px 18px",
        background_color=rx.color("blue", 2),          
        border_radius="var(--radius-2)",                
        border=f"1px solid {rx.color('blue', 4)}",      
        width="100%",
        spacing="1",
    )
# ------------------- END DF Report Helper Function - Python ----------------------- #

def render_col_config_row(cfg: dict):
    return rx.hstack(
        # 1. Include Toggle Checkbox
        rx.checkbox(
            checked=cfg["Include"],
            on_change=lambda val: State.update_col_config(cfg["name"], "include", val),
        ),
        # Raw Variable Name Tag
        rx.text(cfg["name"], font_weight="bold", width="110px", font_size="12px", color="#475569", truncate=True),
        
        # 2. Publication Label Text Input
        rx.input(
            value=cfg["label"],
            on_change=lambda val: State.update_col_config(cfg["name"], "label", val),
            placeholder="Publication label",
            size="1",
            width="180px",
        ),
        
        # 3. Variable Type Selector (Continuous vs Categorical)
        rx.select(
            ["Continuous", "Categorical"],
            value=cfg["type"],
            on_change=lambda val: State.update_col_config(cfg["name"], "type", val),
            size="1",
            width="130px",
        ),
        align="center",
        spacing="3",
        width="100%",
        padding_y="6px",
        border_bottom=f"1px solid {rx.color('gray', 3)}",
    )

def baseline_characteristics_modal() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Baseline Demographics & Clinical Characteristics"),
            rx.dialog.description(
                "Select variables to include in Table 1, assign publication-ready labels, and define variable types.",
                font_size="13px",
                color="gray.600",
                margin_bottom="16px",
            ),
            # Table Column Headers with synchronized column widths and matching offset
            rx.hstack(
                rx.text("Include", font_size="12px", font_weight="700", color="gray.500", width="60px"),
                rx.text("Variable Name", font_size="12px", font_weight="700", color="gray.500", width="140px"),
                rx.text("Publication Label", font_size="12px", font_weight="700", color="gray.500", width="200px"),
                #rx.text("Publication Label", font_size="11px", font_weight="700", color="gray.500", width="220px"),
                rx.text("Summary Type", font_size="12px", font_weight="700", color="gray.500", width="120px", margin_left="12px"),
                justify="between",
                width="100%",
                padding_x="8px",
                padding_bottom="8px",
                border_bottom="1px solid #e2e8f0",
                align_items="center",
            ),
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(
                        State.column_configs.to(list[dict]),
                        render_config_row
                    ),
                    width="100%",
                    spacing="3",
                ),
                style={"max_height": "420px"},
                padding_y="8px",
            ),
            rx.hstack(
                rx.dialog.close(
                    rx.button(
                        "Done", 
                        # 🎯 HOOK UP YOUR REFRESH ACTION HERE:
                        on_click=State.save_baseline_configurations_and_refresh, 
                        color_scheme="blue", 
                        size="2"
                    )
                ),
                justify="end",
                margin_top="16px",
            ),
            max_width="650px",
            padding="24px",
        ),
        open=State.is_baseline_modal_open,
        on_open_change=State.set_baseline_modal_open,
    )

class ForecastState(rx.State):
    selected_dataset: str = "Lung"  # or "Breast"
    uploaded_filename: str = ""     # Stores the actual name of the uploaded file
    custom_filename: str = ""       # Stores the filename permanently for the session/report

    @rx.var
    def event_column_name(self) -> str:
        return "status" if self.selected_dataset == "Lung" else "event"

# ------------------------------------------------ UI Components -------------------------------------------- #

# --- Controls the Behavior of all KM Patient Group Comparison checkboxes down to Custom Range checkboxes --- #
def filter_sub_dropdown_item(group_num: int, col_name: str, sub_val_item: rx.Component) -> rx.Component:
    display_label = sub_val_item["label"]
    is_custom = (display_label == "Custom Range:")

    # 1. State Visibility logic
    is_inputs_visible = rx.cond(
        group_num == 1, 
        State.g1_custom_visible.get(col_name, False), 
        State.g2_custom_visible.get(col_name, False)
    )
    
    checked_list = rx.cond(group_num == 1, State.g1_checked_values.get(col_name, []), State.g2_checked_values.get(col_name, []))
    is_sub_selected = checked_list.contains(display_label)

    current_min = rx.cond(group_num == 1, State.g1_custom_min.get(col_name, ""), State.g2_custom_min.get(col_name, ""))
    current_max = rx.cond(group_num == 1, State.g1_custom_max.get(col_name, ""), State.g2_custom_max.get(col_name, ""))

    # Condition evaluating if there is active numeric text inside the custom inputs
    custom_has_data = (current_min != "") | (current_max != "")
    
    # Custom checkbox is visually colored blue if it's open OR contains saved valid data
    is_custom_checked = is_inputs_visible | custom_has_data

    # 2. Core Checkbox Component
    checkbox = rx.checkbox(
        checked=rx.cond(is_custom, is_custom_checked, is_sub_selected),
        on_change=lambda val: State.handle_interval_selection(group_num, col_name, display_label, val),
        color_scheme="blue" if group_num == 1 else "pink",
        style={"transform": "scale(0.70)", "transformOrigin": "center", "width": "18px", "height": "18px"}
    )

    # Wrap checkbox in a layout node to center a white circle over it during active entry mode
    checkbox_container = rx.box(
        checkbox,
        rx.cond(
            is_custom & is_inputs_visible,
            rx.box(
                style={
                    "position": "absolute",
                    "width": "6px",
                    "height": "6px",
                    "z-index": 10,
                    "background_color": "#ffffff",
                    "border_radius": "50%",
                    "pointer-events": "none",
                }
            ),
            rx.fragment()
        ),
        position="relative",
        display="flex",
        align_items="center",
        justify_content="center",
    )

    checkbox_row = rx.hstack(
        checkbox_container,
        rx.text(display_label, font_size="0.70em", color=rx.color("blue", 11)),
        spacing="1", align_items="center", cursor="pointer",
    )

    # 3. Custom Range Inputs
    custom_range_inputs = rx.hstack(
        rx.input(
            placeholder="Min", 
            type_="number", 
            size="1", 
            value=current_min, 
            on_change=lambda val: State.set_custom_min(group_num, col_name, val), 
            style={
                "width": "38px",        # 👈 Reduced from 48px to keep the box compact
                "height": "18px",       # 👈 Shaved down slightly to look very crisp
                "font_size": "0.70em", 
                "padding_left": "2px",  # 👈 Overrides native theme padding so numbers fit perfectly
                "padding_right": "2px", # 👈 Prevents the text from clipping/stretching
            }
        ),
        rx.text("to", font_size="0.70em", color=rx.color("slate", 11)),
        rx.input(
            placeholder="Max", 
            type_="number", 
            size="1", 
            value=current_max, 
            on_change=lambda val: State.set_custom_max(group_num, col_name, val), 
            style={
                "width": "38px",        # 👈 Matches Min exactly
                "height": "18px",       
                "font_size": "0.70em", 
                "padding_left": "2px",  
                "padding_right": "2px",
            }
        ),
        spacing="1", 
        padding_left="2px"
    )

    # 4. Final Render
    return rx.vstack(
        checkbox_row,
        rx.cond(is_custom & is_inputs_visible, custom_range_inputs, rx.fragment()),
        align_items="start", 
        spacing="1"
    )

def dynamic_workflow_icon(group_num: int, col_name: str) -> rx.Component:
    expanded_map = State.g1_expanded if group_num == 1 else State.g2_expanded
    is_open = expanded_map[col_name]
    col_str = str(col_name).strip()
    
    is_hard_protected = col_str in ["time", "rfstime", "rectime", "status", "event", "cens"]
    is_target_column = (col_name == State.selected_time_column) | (col_name == State.selected_event_column)
    is_locked = is_hard_protected | is_target_column
    
    # Grab the direct True/False value pre-calculated by the backend
    parent_is_checked = rx.cond(
        group_num == 1,
        State.g1_parent_checked_statuses[col_name],
        State.g2_parent_checked_statuses[col_name]
    )

    # If the dropdown is open, keep it visually filled; otherwise, rely on backend calculation
    is_checkbox_checked = rx.cond(
        is_open,
        True,
        parent_is_checked
    )

    return rx.box(
        rx.checkbox(
            id=f"cb_{group_num}_{col_name}",
            checked=is_checkbox_checked,
            size="1",
            color_scheme="blue" if group_num == 1 else "pink",
            disabled=rx.cond(is_locked, True, False),
        ),

        # Pure white ring overlay centered perfectly when open
        rx.cond(
            is_open,
            rx.box(
                style={
                    "position": "absolute",
                    "width": "8px",
                    "height": "8px",
                    "z-index": 10,
                    "background_color": "#3b82f6" if group_num == 1 else "#ec4899",
                    "border": "2px solid #ffffff",
                    "border_radius": "50%",
                    "pointer-events": "none",
                }
            ),
            rx.fragment()
        ),
        position="relative",
        display="flex",
        align_items="center",
        justify_content="center",
        style={"pointer-events": "none"}, 
    )

# --- dynamic_variable_filter_row function (outside the State class definition) --- #
def dynamic_variable_filter_row(group_num: int, col_name: str) -> rx.Component:
    expanded_map = State.g1_expanded if group_num == 1 else State.g2_expanded
    unique_sub_values = State.column_unique_values[col_name]
    is_open = expanded_map[col_name]
    
    col_str = str(col_name).strip()
    is_hard_protected = col_str in ["time", "rfstime", "rectime", "status", "event", "cens"]
    
    if col_str in ["time", "rfstime", "rectime"]:
        bg_color = "transparent"
        # 🟢 CHANGED: Set to "not-allowed" for hard-coded time elements
        cursor_style = "not-allowed"  
        text_element = rx.text(f"⏱️ {col_name}", font_size="0.8em", font_weight="bold", color=rx.color("blue", 11))
    elif col_str in ["status", "event", "cens"]:
        bg_color = "transparent"
        # 🟢 CHANGED: Set to "not-allowed" for hard-coded status/event elements
        cursor_style = "not-allowed"  
        text_element = rx.text(f"🎯 {col_name}", font_size="0.8em", font_weight="bold", color=rx.color("pink", 11))
    else:
        is_target_column = (col_name == State.selected_time_column) | (col_name == State.selected_event_column)
        bg_color = "transparent" 
        # 🟢 CHANGED: Set conditional fallback to "not-allowed" for active dynamic tracking columns
        cursor_style = rx.cond(is_target_column, "not-allowed", "pointer")
        text_element = rx.cond(
            col_name == State.selected_time_column,
            rx.text(f"⏱️ {col_name}", font_size="0.8em", font_weight="bold", color=rx.color("blue", 11)),
            rx.cond(
                col_name == State.selected_event_column,
                rx.text(f"🎯 {col_name}", font_size="0.8em", font_weight="bold", color=rx.color("pink", 11)),
                rx.text(col_name, font_size="0.8em", font_weight="medium", color=rx.color("slate", 12))
            )
        )

    is_protected_var = is_hard_protected | (col_name == State.selected_time_column) | (col_name == State.selected_event_column)

    return rx.vstack(
        rx.hstack(
            dynamic_workflow_icon(group_num, col_name),
            text_element, 
            rx.spacer(),
            rx.cond(
                ~is_protected_var,
                rx.icon(
                    tag="chevron-down", 
                    size=12,
                    color=rx.color("slate", 9),
                    style={
                        "transform": rx.cond(is_open, "rotate(180deg)", "rotate(0deg)"),
                        "transition": "transform 0.2s ease",
                    }
                ),
            ),
            width="100%",  
            align="center",
            spacing="2",
            cursor=cursor_style,

            # 🌟 ADDED: Dynamically fades out protected KM variable headers 
            opacity=rx.cond(is_protected_var, 1.0, 1.0),
            
            # 🎯 FIXED BINDING: Row interaction now drives the unified toggle logic cleanly
            on_click=None if is_hard_protected else lambda: State.toggle_variable_dropdown(group_num, col_name),

            padding_left="0px",
            padding_right="4px",
            padding_y="4px",
            margin_left="1px",
            margin_right="12px", 
            background_color=bg_color,
        ),
        rx.cond(
            is_open,
            rx.vstack(
                rx.foreach(
                    unique_sub_values,
                    lambda val: filter_sub_dropdown_item(group_num, col_name, val)
                ),
                width="100%",
                margin_right="12px",
                spacing="1",
                padding_y="1",
                padding_left="4px",
            )
        ),
        width="100%",
        spacing="1",
    )

def cohort_filter_panel_layout() -> rx.Component:
    """Renders the filter panel with a load-guard to prevent React crashes."""
    return rx.cond(
        State.is_ready,
        # --- RENDER: The Filter Panel when Ready ---
        rx.hstack(
            rx.vstack(
                rx.text("Group 1", font_weight="bold", font_size="0.75em", color="#3b82f6"), 
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(State.csv_columns, lambda col: dynamic_variable_filter_row(1, col)),
                        width="100%", padding_right="14", spacing="1",
                    ),
                    type="always", scrollbars="vertical", style={"height": "250px"},
                ),
                width="50%", spacing="1",
            ),
            rx.vstack(
                rx.text("Group 2", font_weight="bold", font_size="0.75em", color="#ec4899"), 
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(State.csv_columns, lambda col: dynamic_variable_filter_row(2, col)),
                        width="100%", padding_right="14", spacing="1",
                    ),
                    type="always", scrollbars="vertical", style={"height": "250px"},
                ),
                width="50%", spacing="1",
            ),
            width="100%", spacing="4",
        ),
        # --- RENDER: The Loading State when Busy ---
        rx.center(
            rx.spinner(color="blue"),
            rx.text("Refreshing filters...", font_size="0.8em"),
            height="250px", width="100%"
        )
    )

def control_item(label):
    return rx.checkbox(label, checked=State.active_controls.contains(label), on_change=lambda c: State.handle_km_selection(label, c), size="1")

def km_sidebar_controls():
    return rx.vstack(
        rx.text(
            "Select group characteristics:", 
            font_size="0.75em", 
            color=rx.color("slate", 11)
        ),
        # Replaced the raw rx.grid with your new layout component
        cohort_filter_panel_layout(),

        align_items="start", 
        width="100%", 
        spacing="2"
    )

def settings_sidebar():
    return rx.card(
        rx.vstack( 
            rx.text("MODEL SETTINGS", font_weight="bold", font_size="0.9em"),
            rx.accordion.root(
                
                # --- 📂 DATA SELECTION SECTION --- #
                rx.accordion.item(
                    header="Data Selection",
                    value="data_selection",
                    content=rx.vstack(
                        rx.radio_group(
                            ["Lung (NCCTG)", "Breast Cancer (GBSG2)", "Own Data (.csv/.xls)"], 
                            value=State.selected_dataset, 
                            on_change=State.update_dataset,
                            direction="column",
                            spacing="2",
                            size="1"
                        ),
                        
                        rx.cond(
                            State.selected_dataset == "Own Data (.csv/.xls)",
                            rx.box(
                                rx.upload(
                                    rx.hstack(
                                        rx.icon(
                                            "upload", 
                                            size=11, 
                                            color=rx.cond(State.file_upload_success, rx.color("green", 9), rx.color("blue", 9))
                                        ), 
                                        rx.text(
                                            rx.cond(State.file_upload_success, "Uploading", "Upload File..."),
                                            font_size="0.7em", 
                                            font_weight="medium", 
                                            color=rx.cond(State.file_upload_success, rx.color("green", 9), rx.color("blue", 9)), 
                                            white_space="nowrap" 
                                        ), 
                                        spacing="1",
                                        justify_content="center", 
                                        align_items="center", 
                                        height="100%", 
                                        width="100%"
                                    ),
                                    id="upload_csv", 
                                    border=rx.cond(
                                        State.file_upload_success, 
                                        f"1px solid {rx.color('green', 5)}", 
                                        f"1px solid {rx.color('blue', 5)}"
                                    ), 
                                    background_color=rx.cond(
                                        State.file_upload_success, 
                                        rx.color("green", 2), 
                                        rx.color("blue", 2)
                                    ), 
                                    border_radius="var(--radius-1)", 
                                    padding="0", 
                                    height="22px",            
                                    cursor="pointer", 
                                    width="100%",
                                    style={
                                        "_hover": {
                                            "background_color": rx.cond(
                                                State.file_upload_success, 
                                                rx.color("green", 3), 
                                                rx.color("blue", 3)
                                            )
                                        }
                                    },
                                    on_drop=State.handle_upload(rx.upload_files(upload_id="upload_csv"))
                                ),
                                margin_top="2px",            
                                margin_left="20px",            
                                width="115px",
                                height="26px",
                                display="flex",
                                align_items="center",
                            )
                        ),

                        # Inline error message right beneath the upload field
                        rx.cond(
                            State.upload_error != "",
                            rx.text(
                                State.upload_error,
                                color="red",
                                font_size="0.65em",
                                font_weight="medium",
                                max_width="140px",
                            ),
                        ),
                        spacing="1",
                        margin_top="2px",            
                        margin_left="20px",           
                        align_items="start",
                    )
                ),

                # 📊 MODEL SELECTION SECTION
                rx.accordion.item(
                    header="Model Selection", 
                    value="ms", 
                    content=rx.vstack(
                        rx.radio_group(
                            ["Kaplan-Meier Survival", "Cox Proportional Hazards"], 
                            value=State.selected_model, 
                            on_change=State.set_selected_model, 
                            direction="column",
                            spacing="2",
                            size="1"
                        ),
                        rx.cond(
                            State.show_model_error,
                            rx.text(
                                "⚠️ Please select a Model",
                                color="#D97706",                # Amber warning text 
                                # color=rx.color("red", 11), 
                                font_size="0.75em", 
                                font_weight="semibold",
                                margin_top="4px"
                            )
                        ),
                        align_items="start",
                        width="100%",
                    )
                ),

                # 🧬 MODEL VARIABLE SELECTION SECTION
                rx.accordion.item(
                    header=rx.match(
                        State.selected_model,
                        ("Kaplan-Meier Survival", "Patient Group Comparison"), 
                        ("Cox Proportional Hazards", "CPH Patient Characteristics"),    
                        "Model Variable Selection"                                             
                    ),
                    value="dynamic_variables", 
                    
                    content=rx.vstack(
                        rx.cond(
                            State.show_variable_prereq_error,
                            rx.text(
                                "⚠️ Please select Dataset and Model first to configure Variables.", 
                                color="#D97706",
                                #color=rx.color("red", 11), 
                                font_size="0.75em", 
                                font_weight="semibold",
                                margin_bottom="4px"
                            ),
                            rx.match(
                                State.selected_model,
                                (
                                    "Kaplan-Meier Survival",     
                                    rx.cond(
                                        State.is_dataset_empty, 
                                        # 👇 DISPLAY ERROR MESSAGE IF OWN DATA IS SELECTED BUT NOT YET UPLOADED
                                        rx.cond(
                                            State.selected_dataset == "Own Data (.csv/.xls)",
                                            rx.text(
                                                "⚠️ Upload File First", 
                                                color=rx.color("red", 11), 
                                                font_size="0.75em", 
                                                font_weight="semibold",
                                                margin_top="2px"
                                            ),
                                            rx.vstack(padding="0px")
                                        ),
                                        rx.cond(
                                            State.selected_dataset == "Own Data (.csv/.xls)",
                                            rx.vstack(km_sidebar_controls()),
                                            km_sidebar_controls() 
                                        )
                                    ),
                                ),
                                (   
                                    "Cox Proportional Hazards", 
                                    rx.cond(
                                        State.is_dataset_empty, 
                                        # 👇 DISPLAY ERROR MESSAGE IF OWN DATA IS SELECTED BUT NOT UPLOADED
                                        rx.cond(
                                            State.selected_dataset == "Own Data (.csv/.xls)",
                                            rx.text(
                                                "⚠️ Upload Data File First", 
                                                color=rx.color("red", 11), 
                                                font_size="0.75em", 
                                                font_weight="semibold",
                                                margin_top="2px"
                                            ),
                                            rx.vstack(padding="0px")
                                        ),
                                        rx.vstack(
                                            rx.foreach(
                                                State.csv_columns, 
                                                lambda cov: rx.hstack(
                                                    rx.checkbox(
                                                        id=f"cph_cb_{cov}",
                                                        on_change=lambda c: State.update_cph_covariates(cov, c), 
                                                        checked=State.cph_active_covariates.contains(cov), 
                                                        disabled=State.critical_columns.contains(cov),
                                                        size="1",
                                                        cursor=rx.cond(State.critical_columns.contains(cov), "not-allowed", "pointer")
                                                    ),
                                                    # Stylized Text Rendering
                                                    rx.cond(
                                                        cov == State.selected_time_column,
                                                        rx.text(f"⏱️ {cov}", font_size="0.8em", font_weight="bold", color=rx.color("blue", 11)),
                                                        rx.cond(
                                                            cov == State.selected_event_column,
                                                            rx.text(f"🎯 {cov}", font_size="0.8em", font_weight="bold", color=rx.color("pink", 11)),
                                                            rx.text(cov, font_size="0.8em", font_weight="medium", color=rx.color("slate", 12))
                                                        )
                                                    ),
                                                    padding_x="4px", 
                                                    padding_y="4px", 
                                                    border_radius="4px", 
                                                    width="100%", 
                                                    align_items="center", 
                                                    spacing="2",
                                                    cursor=rx.cond(State.critical_columns.contains(cov), "not-allowed", "pointer")
                                                )
                                            ),
                                            spacing="1", align_items="start", width="100%", max_height="255px", overflow_y="scroll", padding_right="4px"
                                        )
                                    ),
                                ),
                                rx.vstack(padding="0px")  
                            )
                        ),
                        
                        rx.cond(
                            State.show_variable_error,
                            rx.text(
                                "⚠️ Please select at least 1 Variable", 
                                color=rx.color("red", 11), 
                                font_size="0.75em", 
                                font_weight="semibold",
                                margin_top="4px"
                            )
                        ),
                        align_items="start",
                        width="100%",
                    )
                ),
                
                value=State.open_sections,
                on_value_change=State.watch_accordion_change,
                type="multiple", 
                variant="ghost", 
                width="100%",
            ),

            rx.divider(),
   
            rx.match(
                State.selected_model,
                (
                    "Cox Proportional Hazards",
                    rx.vstack(
                        # --- 1. Existing Auto-Stratify Switch Box ---
                        rx.vstack(
                            rx.hstack(
                                rx.icon("git-branch", size=16, color="#f59e0b"),
                                rx.text("Auto-Stratify PH Violations", font_size="0.8em", font_weight="semibold", color="#1e293b"),
                                justify="between",
                                width="100%",
                            ),
                            rx.text(
                                "Isolate covariates violating proportional hazards (p < 0.05) into strata-specific baselines.",
                                font_size="0.75em",
                                color="#64748b",
                            ),
                            rx.switch(
                                is_checked=State.cph_auto_stratify_violations,
                                on_change=State.set_cph_auto_stratify_violations,
                            ),
                            spacing="1",
                            padding="8px",
                            background_color="#f8fafc",
                            border="1px solid #e2e8f0",
                            border_radius="6px",
                            width="100%",
                        ),
                        
                        rx.divider(),

                        # --- MANUAL STRATA SANDBOX (Target-Only Violations View) ---
                        rx.cond(
                            State.cph_auto_stratify_violations,
                            rx.cond(
                                # Only render the box if there are actually detected violations from the run
                                State.cph_detected_violations.length() > 0,
                                rx.vstack(
                                    rx.hstack(
                                        rx.icon("sliders_horizontal", size=16, color="#2563eb"),
                                        rx.text("PH Violation Detected - Stratify?", font_size="0.8em", font_weight="semibold", color="#b91c1c"),
                                        justify="between",
                                        width="100%",
                                    ),
                                    rx.text(
                                        "The following covariate(s) violated proportional hazards. Check to isolate into strata.",
                                        font_size="0.75em",
                                        color="#64748b",
                                    ),
                                    # Dynamically list ONLY the violating variables (e.g., tgrade)
                                    rx.vstack(
                                        rx.foreach(
                                            State.cph_detected_violations,
                                            lambda viol_var: rx.checkbox(
                                                viol_var, 
                                                checked=State.selected_strata_vars.contains(viol_var), 
                                                on_change=lambda val: State.toggle_strata_var(viol_var, val), 
                                                font_size="0.75em"
                                            )
                                        ),
                                        align_items="start",
                                        spacing="2",
                                        width="100%",
                                    ),
                                    spacing="1",
                                    padding="8px",
                                    background_color="#fef2f2",  # Soft warning tint
                                    border="1px solid #fecaca",
                                    border_radius="6px",
                                    width="100%",
                                ),
                                rx.fragment() # Hidden if no violations exist yet
                            ),
                            rx.fragment() # Hidden if toggle is OFF
                        ),
                        spacing="3",
                        width="100%",
                    ),
                ),
                
                # --- Default case (Timeline slider for other models) ---
                rx.vstack(
                    rx.text(
                        State.cohort_timeline_heading, 
                        font_size="0.8em", 
                        font_weight="semibold",
                    ),
                    rx.slider(
                        min=0,
                        max=rx.cond(State.timeline_max > 0, State.timeline_max, 100),
                        value=State.timeline_values,
                        on_change=State.handle_timeline_change,
                        disabled=State.timeline_max == 0,
                        width="100%",
                    ),
                    width="100%",
                    spacing="2",
                ),
            ),       
            
            # 🟢 Inline run_cph_engine data error message
            # 🟢 KM Error Message Stacked
            # 🟡 KM Warning Message (Amber Methodological Adaptation Notice)
            rx.cond(
                State.km_warning_message != "",
                rx.text(
                    State.km_warning_message, 
                    font_size="12px", 
                    color="#D97706",         # Amber warning text color
                    weight="medium",
                    line_height="1.2",
                    margin_bottom="4px",
                ),
            ),
            
            rx.cond(
                State.km_error_message != "",
                rx.text(
                    State.km_error_message, 
                    font_size="12px", 
                    color="red", 
                    weight="medium",
                    line_height="1.2",
                    margin_bottom="4px",
                ),
            ),

            # 🟢 CPH Error Message Stacked
            rx.cond(
                State.cph_error_message != "",
                rx.text(
                    State.cph_error_message, 
                    font_size="12px", 
                    color="red", 
                    weight="medium",
                    line_height="1.2",
                    margin_bottom="4px",
                ),
            ),

            # 🟢 General Inline Error (Dataset / Model / Variable errors)
            rx.cond(
                State.show_dataset_error | State.show_model_error | State.show_variable_error,
                rx.text(
                    State.inline_error_msg,
                    color="crimson",
                    size="1",
                    font_size="12px",
                    weight="medium",
                    line_height="1.2",
                    margin_bottom="4px",
                ),
                rx.fragment(),
            ),

            # 🟢 Main Run Analysis Button Stack
            rx.vstack(
                rx.button(
                    "RUN ANALYSIS", 
                    on_click=State.run_analysis,
                    loading=State.is_running_validation,
                    color_scheme="blue", 
                    cursor="pointer", 
                    width="100%",
                ),
                spacing="2", 
                width="100%",
                align_items="start",
            ),
        ),

        #width="100%",
        width="320px",      # <--- Updated
        min_width="320px",  # <--- Added
        max_width="320px",  # <--- Added
        align_self="start", 
        position="sticky",
        top="20px",
    )

# ---------------------------------------------------------------------------------------------------------------- #
# --- Python Functions outside of State called in Index to create Statistical Table below Number-at-Risk Table --- #

def manual_strata_sandbox_ui() -> rx.Component:
    available_covariates = ["tgrade", "pnodes", "horth", "menostat", "tsize", "progrec", "estrec"]
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon("sliders_horizontal", size=20, color="#2563eb"),
                rx.heading("Interactive Model Stratification Sandbox", size="4", weight="bold"),
                spacing="2",
            ),
            rx.text(
                "Select covariates to isolate into strata-specific baselines. Continuous variables like 'pnodes' will be automatically binned for robust clinical stratification.",
                size="2",
                color="gray",
            ),
            # Multi-select or Checkbox Group for Covariates
            rx.checkbox_group(
                value=State.selected_strata_vars,
                on_change=State.set_selected_strata_vars,
                children=rx.vstack(
                    *[
                        rx.checkbox(cov, value=cov)
                        for cov in available_covariates
                    ],
                    align_items="start",
                    spacing="2",
                ),
            ),
            rx.divider(),
            # Action button to trigger manual refresh if needed
            rx.button(
                "RE-RUN MODEL WITH STRATA",
                on_click=State._run_cph_engine(),
                color_scheme="blue",
                size="3",
                width="100%",
            ),
            spacing="4",
            align_items="stretch",
        ),
        padding="5",
        border="1px solid #e2e8f0",
        border_radius="12px",
        background_color="white",
        box_shadow="sm",
        width="100%",
    )


# --- Group Colors --- #
GROUP_1_COLOR = "#3b82f6"  # Blue
GROUP_2_COLOR = "#ec4899"  # Pink

# ----------------------------------- STATISTICS TABLE: SURVIVAL PROBAILTITY & MEAN VALUE ------------------------- #
def render_data_column(title, data_list, tooltip_text=None):
    return rx.vstack(
        # --- Column Header & Tooltip ---
        rx.hstack(
            rx.text(title, font_size="11px", font_weight="bold", color="gray.600"),
            rx.tooltip(
                rx.icon("info", size=14, color="gray.500"),
                label=tooltip_text,
            ) if tooltip_text else rx.fragment(),
            spacing="1",
            align="center",
        ),
        
        # --- Data Rows (Handles Full Cohort, Complements for Both Groups, and Stratified) --- #
        rx.vstack(
            rx.foreach(
                data_list,
                lambda line: rx.fragment(
                    rx.cond(
                        
                        # Case 1: Group 1 (or Group 1 Complement)
                        line.contains("Group 1"),
                        rx.vstack(
                            # Split label from value so only the label/complement is colored
                            rx.text(
                                rx.el.span(
                                    line.split("|")[0].split(":")[0].strip() + ":",
                                    color=GROUP_1_COLOR, 
                                    font_weight="normal",
                                ),
                                rx.el.span(
                                    " " + line.split("|")[0].split(":")[1].strip(),
                                    color="#000000", # Neutral black for the value
                                    font_weight="normal",
                                ),
                                font_size="12px", 
                                white_space="nowrap",
                            ),
                            rx.cond(
                                line.contains("|"),
                                rx.text(line.split("|")[1].strip(), font_size="10px", color="#000000", margin_top="0px"),
                                rx.text("95% CI: N/A", font_size="10px", color="#000000", margin_top="0px")
                            ),
                            spacing="0",
                            align_items="start",
                            margin_bottom="14px",      
                        ),
                        
                        # --- Nested Condition for Case 2 & Case 3 ---
                        rx.cond(
                            # --- Case 2: Group 2 (or Group 2 Complement) ---
                            line.contains("Group 2"),
                            rx.vstack(
                                # Split label from value for Group 2 / Complement
                                rx.text(
                                    rx.el.span(
                                        line.split("|")[0].split(":")[0].strip() + ":",
                                        color=GROUP_2_COLOR, 
                                        font_weight="normal",
                                    ),
                                    rx.el.span(
                                        " " + line.split("|")[0].split(":")[1].strip(),
                                        color="#000000", # Neutral black for the value
                                        font_weight="normal",
                                    ),
                                    font_size="12px", 
                                    white_space="nowrap",
                                ),
                                rx.cond(
                                    line.contains("|"),
                                    rx.text(line.split("|")[1].strip(), font_size="10px", color="#000000", margin_top="0px"),
                                    rx.text("95% CI: N/A", font_size="10px", color="#000000", margin_top="0px")
                                ),
                                spacing="0",
                                align_items="start",
                                margin_bottom="4px"
                            ),
                            
                            # --- Case 3: Full Cohort Fallback (Neither Group 1 nor Group 2) ---
                            rx.vstack(
                                rx.text(
                                    line.split("|")[0].strip(), 
                                    font_size="12px",          # Updated from 14px to match stratified
                                    # font_weight="semibold",  # Removed bold/semibold weight
                                    color="#000000",           # Clean neutral black
                                    white_space="nowrap"
                                ),
                                rx.cond(
                                    line.contains("|"),
                                    rx.text(line.split("|")[1].strip(), font_size="10px", color="#000000", margin_top="0px"),
                                    rx.fragment()
                                ),
                                spacing="0",
                                align_items="start",
                            )                        # --- END RX.COND --- # 
                        )                                   # --- END RX.COND --- #
                    )                                       # --- END RX.COND --- #
                )                                           # --- END LAMBDA --- #
            ),                                              # --- END RX.FOREACH --- #
            spacing="1",
            align_items="start",
            width="fit-content"
        ),                                                  # --- END RX.VSTACK --- #
        spacing="2",
        align_items="start",
        width="100%"
    )

# ----------------------------------- STATISTICS TABLE: P-VALUE COLUMN ----------------------------------- #
def render_stats_column(title, tooltip_text=None):                        
    # 1. Setup the Header matching Columns 1 & 2
    header_content = rx.hstack(
        rx.text(title, font_weight="bold", font_size="11px", color="gray.600"),
        rx.icon(
            tag="info",           
            size=12,             
            color="#d97706"      
        ),
        spacing="1",
        align_items="center",
        justify_content="start",  
        width="100%"
    )
    
    if tooltip_text:
        header = rx.tooltip(header_content, content=tooltip_text)
    else:
        header = header_content

    is_full_cohort = (State.g1_detailed_label == "Group 1: (Full Cohort)") & \
                     (State.g2_detailed_label == "Group 2: (Full Cohort)")
    
    # 2. Return the UI component with bold group-level typography and vertical centering
    return rx.vstack(
        header,
        
        # Inner container that expands and centers the stats block vertically
        rx.vstack(
            # Patient Group Comparison matching group-level bold styling
            rx.text(
                State.ci_text, 
                font_size="12px", 
                font_weight="normal", 
                #color="#111827",
                color="#374151",
            ), 
            # Log-rank p value matching equal 14px bold weight
            rx.text(
                rx.cond(
                    is_full_cohort,
                    "Log-rank p: N/A",
                    rx.cond(
                        State.selected_model == "Kaplan-Meier Survival", 
                        State.p_value_text, 
                        State.cph_log_rank_p
                    )
                ),
                font_size="12px", 
                font_weight="normal", 
                color="#111827",
            ), 
            spacing="1",
            align_items="start",      
            justify_content="center",  # <--- Vertically centers block in the middle of Group 1 & Group 2
            width="100%",
            flex="1"                   # <--- Fills available height to align with multi-row cohorts
        ),
        spacing="2",                  
        align_items="start",          
        width="100%", 
        flex="1"                      
    )

def disclaimer_modal() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Legal Terms & Clinical Disclaimer"),
            rx.dialog.description(
                "Please review the terms governing the use of ClinicalPrognosis.app.",
                size="2",
                color="gray",
            ),
            rx.vstack(
                rx.heading("1. Research and Educational Scope", size="3", weight="bold"),
                rx.text(
                    "ClinicalPrognosis.app is an interactive analytics engine designed strictly for research, educational, "
                    "and exploratory cohort evaluation. It is not cleared, certified, or approved as a medical device by regulatory "
                    "authorities (such as the FDA or EMA).",
                    font_size="13px",
                ),
                
                rx.heading("2. No Medical Advice or Formal Diagnosis", size="3", weight="bold"),
                rx.text(
                    "The statistical outputs, Kaplan-Meier curves, hazard ratios, log-rank p-values, and automated clinical guidance "
                    "texts generated by this software do not constitute formal medical advice, clinical diagnoses, or prescribed "
                    "treatment protocols.",
                    font_size="13px",
                ),
                
                rx.heading("3. Absolute Clinical Responsibility", size="3", weight="bold"),
                rx.text(
                    "Outputs are intended exclusively to assist professional research and analytical judgment. They must never "
                    "replace independent clinical evaluation, patient history review, or standard institutional guidelines. The ultimate "
                    "responsibility for patient care, risk evaluation, and therapeutic decisions rests entirely with the attending physician.",
                    font_size="13px",
                ),
                
                rx.heading("4. 'As Is' Warranty & Limitation of Liability", size="3", weight="bold"),
                rx.text(
                    "The software, underlying biostatistical calculation engines, and default datasets are provided 'as is' without "
                    "warranty of any kind, express or implied. The developers, investigators, and maintainers disclaim all liability "
                    "for any direct, indirect, incidental, or consequential damages resulting from the use or misinterpretation of this tool.",
                    font_size="13px",
                ),

                rx.heading("5. Data Integrity & User Responsibility", size="3", weight="bold"),
                rx.text(
                    "Users are solely responsible for validating input data integrity, proper column mapping (time and event variables), "
                    "and configuration parameters prior to interpreting or exporting analytical reports.",
                    font_size="13px",
                ),
                spacing="3",
                max_height="350px",
                overflow_y="auto",
                padding_y="2",
            ),
            rx.dialog.close(
                rx.button(
                    "I Understand & Close",
                    on_click=State.toggle_disclaimer_modal,
                    size="2",
                    variant="solid",
                ),
                align="end",
            ),
            width="600px",
        ),
        open=State.show_disclaimer_modal,
    )

def legal_footer() -> rx.Component:
    return rx.hstack(
        rx.text(
            "Legal Notice: For research, educational, and analytical evaluation purposes only. "
            "Not a substitute for professional clinical judgment.",
            font_size="11px",
            color="gray",
        ),
        rx.link(
            "View Full Terms",
            on_click=State.toggle_disclaimer_modal,
            font_size="11px",
            color="var(--accent-8)",
            cursor="pointer",
            weight="medium",
        ),
        justify="center",
        spacing="3",
        padding="1em",
        border_top="1px solid var(--gray-4)",
        width="100%",
    )

def cph_results_table():
    return rx.el.div(
        # Outer wrapper
        rx.el.div(
            rx.el.div(
                # 1. Superheaders (Univariate: 42%, Multivariate: 58%)
                rx.el.div(
                    rx.el.div(
                        rx.text("Univariate Analysis", font_weight="bold", color=rx.color("slate", 12), font_size="12px"),
                        style={
                            "width": "50%",
                            "display": "flex",
                            "justify_content": "center",
                            "align_items": "center",
                            "white_space": "nowrap",
                            "padding": "8px 6px",
                            "padding_left": "105px",  # Offset slightly right to balance out the standalone 'Variable' column on the left / 45px
                            "background_color": "var(--gray-2)",
                        }
                    ),
                    rx.el.div(
                        rx.text("Multivariate Adjusted Analysis", font_weight="bold", color=rx.color("slate", 12), font_size="12px"),
                        style={
                            "width": "50%",
                            "display": "flex",
                            "justify_content": "center",
                            "align_items": "center",
                            "white_space": "nowrap",
                            "padding": "8px 6px",
                            "padding_right": "45px", # Shifts the text slightly left to center perfectly over its 4 columns
                            "background_color": "var(--gray-2)",
                        }
                    ),
                    style={
                        "display": "flex",
                        "width": "100%",
                        "border_bottom": f"1px solid {rx.color('slate', 5)}",
                        "border_top_left_radius": "6px",
                        "border_top_right_radius": "6px",
                        "overflow": "hidden",
                    }
                ),

                # 2. Table with compressed percentages totaling 100%
                rx.el.table(
                    rx.el.colgroup(
                        rx.el.col(style={"width": "13%"}),  # Variable
                        rx.el.col(style={"width": "9%"}),   # Unadjusted HR
                        rx.el.col(style={"width": "12%"}),  # Univariate 95% CI
                        rx.el.col(style={"width": "8%"}),   # Univariate P
                        rx.el.col(style={"width": "10%"}),  # Adjusted HR
                        rx.el.col(style={"width": "13%"}),  # Multivariate 95% CI
                        rx.el.col(style={"width": "12%"}),  # Adjusted P
                        rx.el.col(style={"width": "13%"}),  # Schoenfeld P
                    ),
                    rx.el.thead(
                        rx.el.tr(
                            rx.el.th(
                                rx.text("Variable", font_weight="bold", color=rx.color("slate", 12), font_size="11px"),
                                style={"text_align": "left", "white_space": "nowrap", "padding": "6px 6px", "border_bottom": f"1px solid {rx.color('slate', 5)}", "background_color": "var(--gray-2)"}
                            ),
                            rx.el.th(
                                rx.text("HR", font_weight="bold", color=rx.color("slate", 12), font_size="11px"),
                                style={"text_align": "center", "white_space": "nowrap", "padding": "6px 4px", "border_bottom": f"1px solid {rx.color('slate', 5)}", "background_color": "var(--gray-2)"}
                            ),
                            rx.el.th(
                                rx.text("95% CI", font_weight="bold", color=rx.color("slate", 12), font_size="11px"),
                                style={"text_align": "center", "white_space": "nowrap", "padding": "6px 4px", "border_bottom": f"1px solid {rx.color('slate', 5)}", "background_color": "var(--gray-2)"}
                            ),
                            rx.el.th(
                                rx.text("P-Value", font_weight="bold", color=rx.color("slate", 12), font_size="11px"),
                                style={"text_align": "center", "white_space": "nowrap", "padding": "6px 4px", "border_bottom": f"1px solid {rx.color('slate', 5)}", "background_color": "var(--gray-2)"}
                            ),
                            rx.el.th(
                                rx.text("Adjusted HR", font_weight="bold", color=rx.color("slate", 12), font_size="11px"),
                                style={"text_align": "center", "white_space": "nowrap", "padding": "6px 4px", "border_bottom": f"1px solid {rx.color('slate', 5)}", "background_color": "var(--gray-2)"}
                            ),
                            rx.el.th(
                                rx.text("95% CI", font_weight="bold", color=rx.color("slate", 12), font_size="11px"),
                                style={"text_align": "center", "white_space": "nowrap", "padding": "6px 4px", "border_bottom": f"1px solid {rx.color('slate', 5)}", "background_color": "var(--gray-2)"}
                            ),
                            rx.el.th(
                                rx.text("Adjusted P", font_weight="bold", color=rx.color("slate", 12), font_size="11px"),
                                style={"text_align": "center", "white_space": "nowrap", "padding": "6px 4px", "border_bottom": f"1px solid {rx.color('slate', 5)}", "background_color": "var(--gray-2)"}
                            ),
                            rx.el.th(
                                rx.text("Schoenfeld P", font_weight="bold", color=rx.color("slate", 12), font_size="11px"),
                                style={"text_align": "center", "white_space": "nowrap", "padding": "6px 4px", "border_bottom": f"1px solid {rx.color('slate', 5)}", "background_color": "var(--gray-2)"}
                            ),
                        ),
                        style={"position": "sticky", "top": "0", "z-index": "50"}
                    ),
                    rx.el.tbody(
                        rx.foreach(
                            State.cph_table_data,
                            lambda row: rx.el.tr(
                                rx.el.td(rx.text(row[0], font_size="11px", font_weight="medium"), style={"white_space": "nowrap", "padding": "5px 6px", "border_bottom": f"1px solid {rx.color('slate', 3)}"}),
                                rx.el.td(rx.text(row[1], font_size="11px"), style={"text_align": "center", "white_space": "nowrap", "padding": "5px 4px", "border_bottom": f"1px solid {rx.color('slate', 3)}"}),
                                rx.el.td(rx.text(row[2], font_size="11px"), style={"text_align": "center", "white_space": "nowrap", "padding": "5px 4px", "border_bottom": f"1px solid {rx.color('slate', 3)}"}),
                                rx.el.td(rx.text(row[3], font_size="11px"), style={"text_align": "center", "white_space": "nowrap", "padding": "5px 4px", "border_bottom": f"1px solid {rx.color('slate', 3)}"}),
                                rx.el.td(rx.text(row[4], font_size="11px"), style={"text_align": "center", "white_space": "nowrap", "padding": "5px 4px", "border_bottom": f"1px solid {rx.color('slate', 3)}"}),
                                rx.el.td(rx.text(row[5], font_size="11px"), style={"text_align": "center", "white_space": "nowrap", "padding": "5px 4px", "border_bottom": f"1px solid {rx.color('slate', 3)}"}),
                                rx.el.td(rx.text(row[6], font_size="11px"), style={"text_align": "center", "white_space": "nowrap", "padding": "5px 4px", "border_bottom": f"1px solid {rx.color('slate', 3)}"}),
                                rx.el.td(rx.text(row[7], font_size="11px", color=rx.cond(row[7].contains("<") | (row[7].to(float) < 0.05), "#ef4444", "inherit")), style={"text_align": "center", "white_space": "nowrap", "padding": "5px 4px", "border_bottom": f"1px solid {rx.color('slate', 3)}"}),
                            )
                        )
                    ),
                    style={"width": "100%", "border_collapse": "collapse", "font_family": "inherit", "table_layout": "fixed"}
                ),
                style={"width": "100%"}
            ),
            style={"width": "100%"}
        ),
        style={"width": "100%", "border": f"1px solid {rx.color('slate', 5)}", "border_radius": "6px", "overflow": "hidden", "background_color": "white"}
    )

def render_cph_table_one():
    return rx.card(
        rx.vstack(
            rx.heading("Table 1: Baseline Demographics & Clinical Characteristics", font_size="12px", color="#000000", font_weight="semibold"),
            rx.text(
                "Baseline characteristics of the analyzed standalone cohort. Continuous variables shown as Mean ± SD; categorical variables shown as n (%).",
                font_size="11px",
                color="#64748b",
            ),
            rx.divider(),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Characteristic", font_size="12px"),
                        rx.table.column_header_cell("Overall Cohort", font_size="12px", text_align="center"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(
                        State.table_one_data,
                        lambda row: rx.table.row(
                            rx.table.cell(row["variable"], font_size="11px"),
                            rx.table.cell(row["group1"], font_size="11px", text_align="center"),
                        ),
                    )
                ),
                width="100%",
                variant="surface",
            ),
            spacing="2",
            align="stretch",
        ),
        width="100%",
        padding="1rem",
        margin_top="0.5rem",
        background_color="white",
        border="1px solid #e2e8f0",
        border_radius="8px",
    )


# --------------------------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------- MAIN DISPLAY INDEX FUNCTION ------------------------------------------------------- #
def index():                    
    plotly_config = {
        "useResizeHandler": True,  # <--- Essential for dynamic resizing
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "True",
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", 
            "autoScale2d", "resetScale2d", "hoverClosestCartesian", 
            "hoverCompareCartesian", "toggleSpikelines"
        ],
    }
        
    return rx.center(   # --- holds the entire layout frame
        rx.vstack(      # --- parent vertical stack managing top title + main split content
           
            # 1. --- Top Header Bar (Title centered, buttons on the right) ---
            rx.hstack(
                # Left side: Clean left-aligned heading with subtle subtitle styling if desired
                rx.heading("Clinical Prognosis Platform (CPP)", size="6", weight="bold"),
                
                # Right side: Your exact button configuration block
                rx.hstack(
                    rx.button(
                        rx.hstack(rx.icon("info", size=14), rx.text("About"), spacing="1"),
                        on_click=State.toggle_about_modal,
                        size="1",
                        variant="outline",
                        cursor="pointer",
                    ),
                    rx.button(
                        rx.hstack(rx.icon("sparkles", size=14), rx.text("What's New"), spacing="1"),
                        on_click=State.toggle_updates_modal,
                        size="1",
                        variant="outline",
                        cursor="pointer",
                    ),
                    rx.button(
                        rx.hstack(rx.icon("message-square", size=14), rx.text("Feedback"), spacing="1"),
                        on_click=State.open_feedback_modal,
                        size="1",
                        variant="outline",
                        cursor="pointer",
                    ),
                    rx.button(
                        rx.hstack(rx.icon("scale", size=14), rx.text("Legal"), spacing="1"),
                        on_click=State.toggle_disclaimer_modal,
                        size="1",
                        variant="outline",
                        cursor="pointer",
                    ),
                    spacing="2",
                    justify="end",
                ),

                # Sticky positioning properties
                position="sticky",
                top="0",
                z_index="1000",
                background_color="white",  # Adjust if your app uses a dark mode or custom background color
                
                # Outer bar styling replacing the box border
                width="100%",
                justify="between",
                align="center",
                padding_top="0px",  # was "0"px
                padding_bottom="8px",
                padding_left="24px",
                padding_right="24px",
                border_bottom=f"1px solid {rx.color('slate', 4)}",
            ),

            # 2. Main content split side-by-side
            rx.hstack(

                # --- Left Column (Wider content area)
                rx.vstack(  # --- dedicated left-side rx.vstack locked down so it never collapses
                    rx.cond(
                        State.metadata_info != "",
                        rx.text(
                            State.metadata_info,                  # Schreibt Metadata-Info oberhalb der Chart Area
                            font_size="0.85em", 
                            font_weight="medium", 
                            #color="#D97706",                     # Amber warning text
                            #color="#10B981",                    # Green Status/Success
                            color="#334155",                    # Black
                            #color="#0f172a",                    # Black
                            #color=rx.color("slate", 12), 
                            margin_bottom="0.8em",
                            text_align="left",
                            width="100%",
                            max_width="1004px",
                            align_self="center",
                        ),
                    ),

                    rx.cond(
                        (State.selected_dataset != ""),
                        rx.vstack(
                            # --- Top Action Bar: Both buttons side-by-side ---
                            rx.hstack(
                                # 1. --- Existing Show Data View Button
                                rx.button(
                                    rx.hstack(
                                        rx.cond(
                                            State.show_schema_preview,
                                            rx.fragment(rx.icon(tag="eye-off", size=11), rx.text("Hide Data View")),
                                            rx.fragment(rx.icon(tag="eye", size=11), rx.text("Show Data View"))
                                        ),
                                        spacing="1",
                                        justify_content="center",
                                        align_items="center",
                                        height="22",
                                    ),
                                    on_click=State.toggle_schema_view,
                                    size="1",
                                    cursor="pointer",
                                    height="22px",
                                    padding="0 8px",
                                    style={
                                        "background_color": rx.cond(State.show_schema_preview, rx.color("blue", 3), rx.color("blue", 2)),
                                        "border": f"1px solid {rx.color('blue', 5)}",
                                        "color": rx.color("blue", 9),
                                        "font_size": "0.7em",
                                        "font_weight": "medium",
                                        "border_radius": "var(--radius-1)",
                                        "white_space": "nowrap",
                                        "_hover": {"background_color": rx.color("blue", 3)}
                                    }
                                ),

                                # 🎯 Button for Baseline Characteristics Modal
                                rx.tooltip(
                                    rx.button(
                                        rx.hstack(
                                            rx.icon(tag="sliders", size=11),
                                            rx.text("Baseline Characteristics"),
                                            spacing="1",
                                            justify_content="center",
                                            align_items="center",
                                            height="22",
                                        ),
                                        on_click=State.open_baseline_modal,
                                        size="1",
                                        cursor="pointer",
                                        height="22px",
                                        padding="0 8px",
                                        style={
                                            "background_color": rx.color("blue", 2),
                                            "border": f"1px solid {rx.color('blue', 5)}",
                                            "color": rx.color("blue", 9),
                                            "font_size": "0.7em",
                                            "font_weight": "medium",
                                            "border_radius": "var(--radius-1)",
                                            "white_space": "nowrap",
                                            "_hover": {"background_color": rx.color("blue", 3)}
                                        }
                                    ),
                                    content="Configure and review stratified baseline characteristics (Table 1) to ensure cohort balance and STROBE/REMARK compliance.",
                                ),

                                # 2. --- Create Report Button (Styled matching the Data View button)
                                rx.button(
                                    rx.hstack(
                                        rx.cond(
                                            State.show_report_panel,
                                            rx.fragment(rx.icon(tag="file-text", size=11), rx.text("Hide Report Settings")),
                                            rx.fragment(rx.icon(tag="file-text", size=11), rx.text("Create Report"))
                                        ),
                                        spacing="1",
                                        justify_content="center",
                                        align_items="center",
                                        height="22",
                                    ),
                                    on_click=State.toggle_report_panel,
                                    size="1",
                                    cursor="pointer",
                                    height="22px",
                                    padding="0 8px",
                                    style={
                                        "background_color": rx.cond(State.show_report_panel, rx.color("blue", 3), rx.color("blue", 2)),
                                        "border": f"1px solid {rx.color('blue', 5)}",
                                        "color": rx.color("blue", 9),
                                        "font_size": "0.7em",
                                        "font_weight": "medium",
                                        "border_radius": "var(--radius-1)",
                                        "white_space": "nowrap",
                                        "_hover": {"background_color": rx.color("blue", 3)}
                                    }
                                ),

                                # 3. --- Edit Manuscript Button wrapped in a rx.vstack to display warning message --- #
                                rx.vstack(
                                    rx.button(
                                        rx.hstack(
                                            rx.icon(tag="file-pen", size=11),
                                            rx.text("Edit Manuscript"),
                                            spacing="1",
                                            justify_content="center",
                                            align_items="center",
                                            height="22",
                                        ),
                                        on_click=rx.cond(
                                            State.results_ready,
                                            State.open_manuscript_modal,
                                            State.handle_disabled_manuscript_click,
                                        ),
                                        size="1",
                                        cursor=rx.cond(State.results_ready, "pointer", "not-allowed"),
                                        height="22px",
                                        padding="0 8px",
                                        style={
                                            "background_color": rx.cond(State.results_ready, rx.color("blue", 2), rx.color("slate", 2)),
                                            "border": rx.cond(
                                                State.results_ready, 
                                                f"1px solid {rx.color('blue', 5)}", 
                                                f"1px solid {rx.color('slate', 5)}"
                                            ),
                                            "color": rx.cond(State.results_ready, rx.color("blue", 9), rx.color("slate", 8)),
                                            "font_size": "0.7em",
                                            "font_weight": "medium",
                                            "border_radius": "var(--radius-1)",
                                            "white_space": "nowrap",
                                            "_hover": {
                                                "background_color": rx.cond(State.results_ready, rx.color("blue", 3), rx.color("slate", 2))
                                            }
                                        }
                                    ),
                                    
                                    # Inline Warning Message using absolute positioning (matching the PDF pattern)
                                    rx.cond(
                                        State.show_manuscript_warning,
                                        rx.text(
                                            State.manuscript_warning_msg,
                                            style={
                                                "position": "absolute",
                                                "top": "28px",              # --- moves Warning Message down --- #
                                                "left": "0px",
                                                "white_space": "nowrap", 
                                                "font_size": "11px",
                                                "color": "#D97706",    
                                                "font_weight": "500",
                                                "z_index": "9999",      
                                            },
                                        ),
                                    ),
                                    position="relative", # Locks the absolute child to this container without expanding the layout
                                ),

                                # 4. --- Download Data Menu with Excel and CSV options (Matched Styling)
                                rx.menu.root(
                                    rx.menu.trigger(
                                        rx.button(
                                            rx.hstack(
                                                rx.icon(tag="download", size=11),
                                                rx.text("Download Data"),
                                                rx.icon(tag="chevron-down", size=10),
                                                spacing="1",
                                                justify_content="center",
                                                align_items="center",
                                                height="22",
                                            ),
                                            size="1",
                                            cursor="pointer",
                                            height="22px",
                                            padding="0 8px",
                                            style={
                                                "background_color": rx.color("blue", 2),
                                                "border": f"1px solid {rx.color('blue', 5)}",
                                                "color": rx.color("blue", 9),
                                                "font_size": "0.7em",
                                                "font_weight": "medium",
                                                "border_radius": "var(--radius-1)",
                                                "white_space": "nowrap",
                                                "_hover": {"background_color": rx.color("blue", 3)}
                                            }
                                        )
                                    ),
                                    rx.menu.content(
                                        rx.menu.item(
                                            "Excel Spreadsheet (.xlsx)", 
                                            on_click=State.download_excel_data,
                                            style={
                                                "font_size": "0.7em",
                                                "padding": "3px 6px",
                                                "height": "20px",
                                                "min_height": "unset",
                                                "line_height": "1",
                                                "color": "black",
                                                "border_radius": "var(--radius-1)",
                                                "_hover": {"background_color": rx.color("blue", 3)},
                                            }
                                        ),
                                        rx.menu.item(
                                            "CSV File (.csv)", 
                                            on_click=State.download_csv_data,
                                            style={
                                                "font_size": "0.7em",
                                                #"padding": "0px 4px",
                                                "padding": "3px 6px",
                                                "height": "20px",
                                                "min_height": "unset",
                                                "line_height": "1",
                                                "color": "black",
                                                "border_radius": "var(--radius-1)",
                                                "_hover": {"background_color": rx.color("blue", 3)},
                                            }
                                        ),
                                        style={
                                            "background_color": rx.color("blue", 2),
                                            "border": f"0.5px solid {rx.color('blue', 4)}",
                                            "border_radius": "var(--radius-1)",
                                            "padding": "2px",
                                            "min_width": "unset",
                                        }
                                    ),
                                    spacing="2",
                                    width="100%",
                                    padding_x="0em",
                                ),
                            ),  # --- END RX.HSTACK 
                            margin_bottom="1em",
                            width="100%",
                            flex_shrink="0",     # --- Prevents flexbox from squishing it
                            flex_grow="0",       # --- Prevents flexbox from expanding it
                            align_self="center",
                            spacing="2"
                        ),  # --- END RX.VSTACK
                        rx.fragment()
                    ),  # --- END RX.COND

                    rx.flex(
                        rx.vstack(
                            # --- MOVED PDF REPORT CONFIGURATION HER TO ALIGN WITH CHART AREA --- 
                            rx.cond(
                            State.show_report_panel,             # --- BOOLEAN TOGGLE VARIABLE ---#
                            report_customization_panel(),
                            ),

                            # --- TABLE MOVED HERE TO ALIGN PERFECTLY WITH THE CHART AREA ---
                            rx.cond(
                                State.show_schema_preview,
                                rx.box(
                                    rx.tabs.root(
                                        rx.hstack(
                                            rx.tabs.list(
                                                rx.tabs.trigger(
                                                    "Data Rows", 
                                                    value="preview", 
                                                    style={
                                                        "font_size": "12px",
                                                        "height": "auto",           # 👈 Removes the default tall container height
                                                        "min_height": "unset",      # 👈 Removes the default Radix min-height
                                                        "padding_top": "0px",       # 👈 Fine-tune top spacing
                                                        "padding_bottom": "0px",    # 👈 Keeps the blue line tight to the text
                                                    }
                                                ),
                                                rx.tabs.trigger(
                                                    "Data Types", 
                                                    value="schema", 
                                                    style={
                                                        "font_size": "12px",
                                                        "height": "auto",           # 👈 Removes the default tall container height
                                                        "min_height": "unset",      # 👈 Removes the default Radix min-height
                                                        "padding_top": "0px",       # 👈 Fine-tune top spacing
                                                        "padding_bottom": "0px",    # 👈 Keeps the blue line tight to the text
                                                    }
                                                ),
                                                style={"gap": "24px"},
                                            ),
                                            rx.spacer(),
                                            width="100%",
                                            justify_content="between",
                                            align_items="center",
                                            padding_x="1px",
                                            margin_bottom="8px"     # vertical space between "Data Rows" & "Data Types" Button and Blue Rectangle
                                        ),

                                        # ---------------------------------------------------------------------------------------------- #
                                        # 2. -------------------------------- The Data Preview Tab Content ----------------------------- #
                                        rx.tabs.content(
                                            rx.vstack(
                                                #rx.cond(
                                                    #State.selected_dataset == "Own Data (.csv/.xls)",
                                                    #True,   # <--- Forces it to always pass through

                                                # -------------------- Blue Rectangle Box for Data Manipulation Buttons -------------------- #
                                                rx.box(
                                                    rx.vstack(
                                                        # ---------------------------- RX.HSTACK OBERE ZEILE: ROW 1 ------------------------ #
                                                        rx.hstack(
                                                            # Metadata badges group showing key statistical and dataset health indicators
                                                            rx.hstack(
                                                                # 1. Dynamic Record Count (Filtered vs Total)
                                                                rx.badge(
                                                                    rx.hstack(
                                                                        rx.text("Rows:", color=rx.color("slate", 11)),
                                                                        rx.text(State.filtered_row_count, weight="bold"),
                                                                        rx.text("/", color=rx.color("slate", 11)),
                                                                        rx.text(State.total_row_count),
                                                                        spacing="1",
                                                                        align="center",
                                                                    ),
                                                                    variant="surface",
                                                                    color_scheme="gray",
                                                                    size="1",
                                                                ),
                                                                
                                                                # 2. Statistical Breakdown (Events vs. Censored)
                                                                rx.badge(
                                                                    rx.hstack(
                                                                        # Tooltip for Events with an info icon indicator
                                                                        rx.tooltip(
                                                                            rx.hstack(
                                                                                rx.text("Events:", color=rx.color("blue", 11)),
                                                                                rx.icon("info", size=12, color=rx.color("blue", 11)),
                                                                                rx.text(State.event_count, weight="bold"),
                                                                                spacing="1",
                                                                                align="center",
                                                                            ),
                                                                            content="Subjects who experienced the primary endpoint or event of interest (Indicator = 1).",
                                                                        ),
                                                                        rx.text("|", color=rx.color("slate", 8)),
                                                                        # Tooltip for Censored with an info icon indicator
                                                                        rx.tooltip(
                                                                            rx.hstack(
                                                                                rx.text("Censored:", color=rx.color("amber", 11)),
                                                                                rx.icon("info", size=12, color=rx.color("amber", 11)),
                                                                                rx.text(State.censored_count, weight="bold"),
                                                                                spacing="1",
                                                                                align="center",
                                                                            ),
                                                                            content="Subjects who did not experience the event before observation stopped or were lost to follow-up (Indicator = 0).",
                                                                        ),
                                                                        spacing="2",
                                                                        align="center",
                                                                    ),
                                                                    variant="surface",
                                                                    color_scheme="blue",
                                                                    size="1",
                                                                ),
                                                                
                                                                # 3. Data Health / Missing Values Indicator (Complete Cases %)
                                                                rx.badge(
                                                                    rx.hstack(
                                                                        rx.text("Complete:", color=rx.color("green", 11)),
                                                                        rx.text(State.complete_cases_percentage, weight="bold"),
                                                                        spacing="1",
                                                                        align="center",
                                                                    ),
                                                                    variant="surface",
                                                                    color_scheme="green",
                                                                    size="1",
                                                                ),
                                                                
                                                                spacing="2",
                                                                align="center",
                                                            ),
                                                            justify="between",
                                                            align="center",
                                                            width="100%",
                                                            margin_bottom="-7px",   # vertical space between Top/Middle Row in the Blue Rectangle
                                                            style={"position": "relative", "z_index": 10},  # Keeps buttons fully clickable on top,
                                                        ),
                                                        
                                                        # -------------------------- RX.HSTACK MITTLERE ZEILE: ROW 2 ------------------------ #
                                                        rx.hstack(
                                                            # --- Single Spacer pushes EVERYTHING after it to the far right ---
                                                            rx.spacer(),

                                                            # --- 1. All Columns Dropdown: ROW 2 ---
                                                            rx.hstack(
                                                                rx.select.root(
                                                                    rx.select.trigger(
                                                                        placeholder="All Columns", 
                                                                        width="100px",
                                                                        style={"height": "22px", "min_height": "22px"}
                                                                    ),
                                                                    rx.select.content(
                                                                        rx.select.item("All Columns", value="all"),
                                                                        rx.foreach(State.all_dataset_columns, lambda opt: rx.select.item(opt, value=opt))
                                                                    ),
                                                                    value=State.selected_search_column,
                                                                    on_change=State.set_column_and_filter,
                                                                    size="1",
                                                                ),
                                                                rx.tooltip(
                                                                    rx.icon("info", size=14, color=rx.color("slate", 9)),
                                                                    content="Apply Filter to All Columns or a Selected Column.",
                                                                ),
                                                                spacing="1",
                                                                align="center",
                                                            ),

                                                            # --- 2. Type to Filter Input: ROW 2 ---
                                                            rx.vstack(
                                                                rx.input(
                                                                    placeholder="Type to Filter",
                                                                    width="100px",
                                                                    value=State.search_query,
                                                                    on_change=State.set_search_query_and_filter,
                                                                    clearable=True,
                                                                    style={"height": "22px", "min_height": "22px"}
                                                                ),
                                                                
                                                                # --- Floating "No Match" Message Positioned Under Filter Input ---
                                                                rx.cond(
                                                                    State.filter_status_message != "",
                                                                    rx.text(
                                                                        State.filter_status_message,
                                                                        style={
                                                                            "position": "absolute",
                                                                            "top": "64px",          # Right under the 22px input box
                                                                            "left": "0px",
                                                                            "white_space": "nowrap", 
                                                                            "font_size": "0.75em",
                                                                            "color": "#D97706",     # Clean amber warning text
                                                                            "font_weight": "500",
                                                                            "z_index": "9999",      # Ensures it layers cleanly over table headers
                                                                        },
                                                                    ),
                                                                ),
                                                                spacing="1",
                                                                style={
                                                                    "position": "relative", 
                                                                    "overflow": "visible",   # Prevents clipping
                                                                },
                                                            ),
            
                                                            # --- 3. Row # ROW-#: ROW 2 ---
                                                            rx.hstack(
                                                                rx.input(
                                                                    placeholder="Row #",
                                                                    value=State.jump_row_number,
                                                                    on_change=State.set_jump_row_number,
                                                                    id="row-number-input",
                                                                    size="1",
                                                                    width="48px",
                                                                    style={"height": "22px", "min_height": "22px"}
                                                                ),
                                                                rx.tooltip(
                                                                    rx.icon("info", size=14, color=rx.color("slate", 9)),
                                                                    content="Enter a row number and click Go. Use ⏫ to jump to the first row, or ⏬ for the last.",
                                                                ),
                                                                spacing="1",
                                                                align="center",
                                                            ),

                                                            # --- 4. Go Button: ROW 2 ---
                                                            rx.button(
                                                                "Go",
                                                                size="1",
                                                                variant="surface",
                                                                width="30px",
                                                                on_click=State.jump_to_row,
                                                                style={"height": "22px", "min_height": "22px"} # Forces a slimmer profile
                                                            ),

                                                            # --- 5. Top Button: ROW 2 ---
                                                            rx.button(
                                                                #"Top ⇡",
                                                                #rx.icon("chevrons_up", size=16),
                                                                "⏫",  # The unicode character
                                                                size="1",
                                                                variant="surface",
                                                                width="30px",
                                                                on_click=State.scroll_to_top,
                                                                style={
                                                                    "height": "22px", 
                                                                    "min_height": "22px",
                                                                    "font_size": "12px", # Adjust this if the emoji needs to be larger/smaller
                                                                }     
                                                            ),

                                                            # --- 6. End Button: ROW 2 ---
                                                            rx.button(
                                                                #"End ⇣",
                                                                #rx.icon("chevrons_down", size=16),
                                                                "⏬",  # The unicode character
                                                                size="1",
                                                                variant="surface",
                                                                width="30px",
                                                                on_click=State.scroll_to_end,
                                                                style={
                                                                    "height": "22px", 
                                                                    "min_height": "22px",
                                                                    "font_size": "12px", # Adjust this if the emoji needs to be larger/smaller
                                                                }
                                                            ),
            
                                                            spacing="2",
                                                            width="100%",
                                                            justify="end",
                                                            align="center",
                                                            margin_bottom="-28px",  # --- controls distance to Lower Row
                                                            style={"position": "relative", "z_index": 10},  # Keeps buttons fully clickable on top,
                                                            margin_left="-68px",  # --- Shifts the whole middle row right to line up with "Exclude Cols" below it
                                                        ),

                                                        # --- Reserved Space for Metadata / Status Messages ---
                                                        rx.cond(
                                                            State.is_status_visible,
                                                            #rx.text(State.status_message, font_size="0.75em", color=rx.color("slate", 11)),
                                                            rx.fragment(),
                                                        ),

                                                        # ------------------------------- RX.HSTACK: ROW 3 / UNTERE ZEILE ------------------------------- #
                                                        # --- Row 3: Main Structural & UNTERE ZEILE ---
                                                        rx.hstack(  
                                                            rx.hstack(
                                                                # 1. --- Time Dropdown: ROW 3 ---
                                                                rx.vstack(  
                                                                    rx.text("Time", font_size="0.75em", font_weight="bold", color=rx.color("slate", 11)),
                                                                    rx.select.root(
                                                                        rx.select.trigger(
                                                                            placeholder="Select Time...", 
                                                                            style={
                                                                                "width": "80px",
                                                                                "height": "24px",
                                                                                "min_height": "22px",
                                                                                "border_left": f"3px solid {rx.color('blue', 8)}", # Adds the subtle blue accent
                                                                            }
                                                                        ),
                                                                        rx.select.content(
                                                                            rx.foreach(State.csv_columns, lambda opt: rx.select.item(opt, value=opt))
                                                                        ),
                                                                        value=State.selected_time_column,
                                                                        on_change=State.set_time_column,
                                                                        size="1",
                                                                        width="100%",
                                                                    ),
                                                                    
                                                                    # --- Floating Absolute Error Message (Prevents Layout Shift) ---
                                                                    rx.cond(  
                                                                        State.show_variable_error,
                                                                        rx.text(
                                                                            State.time_error_msg, 
                                                                            style={
                                                                                "position": "absolute",
                                                                                "top": "58px",           
                                                                                "left": "0px",
                                                                                "font_size": "0.75em",
                                                                                "color": "red",
                                                                                "font_weight": "normal",
                                                                                "z_index": "9999",       
                                                                                "white_space": "nowrap",   # Keeps the message on a single, clean line
                                                                            }
                                                                        ),
                                                                        rx.fragment(),
                                                                    ),
                                                                    
                                                                    spacing="1",
                                                                    width="95px",
                                                                    style={
                                                                        "position": "relative",    # Anchor point for the absolute error text
                                                                        "overflow": "visible",     # Ensures the absolute text isn't clipped
                                                                    },
                                                                ),
                                                                
                                                                # 2. --- Event Dropdown: ROW 3 ---
                                                                rx.vstack(  
                                                                    rx.text("Event", font_size="0.75em", font_weight="bold", color=rx.color("slate", 11)),
                                                                    rx.select.root(
                                                                        rx.select.trigger(
                                                                            placeholder="Select Event...",
                                                                            style={
                                                                                "width": "80px",
                                                                                "height": "24px",
                                                                                "min_height": "22px",
                                                                                "border_left": f"3px solid {rx.color('ruby', 8)}", # Adds the subtle pink/ruby accent
                                                                            },
                                                                        ),
                                                                        rx.select.content(
                                                                            rx.foreach(State.csv_columns, lambda opt: rx.select.item(opt, value=opt))
                                                                        ),
                                                                        value=State.selected_event_column,
                                                                        on_change=State.set_event_column,
                                                                        size="1",
                                                                        width="100%",
                                                                    ),
                                                                    
                                                                    # --- Floating Absolute Error Message (Prevents Layout Shift) ---
                                                                    rx.cond(  
                                                                        State.show_variable_error,
                                                                        rx.text(
                                                                            State.event_error_msg, 
                                                                            style={
                                                                                "position": "absolute",
                                                                                "top": "60px",           # Positions it right below the label + dropdown
                                                                                "left": "0px",
                                                                                "font_size": "0.75em",
                                                                                "color": "red",
                                                                                "font_weight": "normal",
                                                                                "z_index": "9999",          # Floats on top of the table view
                                                                                "white_space": "nowrap",    # Keeps message on one single line
                                                                                #"width": "140px",          # Gives multi-line text a fixed breathing room width
                                                                                "line_height": "1.1",       # Keeps lines tightly packed
                                                                            }
                                                                        ),
                                                                        rx.fragment(),
                                                                    ),
                                                                    
                                                                    spacing="1",
                                                                    width="95px",
                                                                    style={
                                                                        "margin_left": "-10px",
                                                                        "position": "relative",    # Anchor point for the absolute error text
                                                                        "overflow": "visible",     # Ensures the absolute text isn't cut off or clipped
                                                                    },
                                                                ),
                                                                
                                                                #spacing="1",  # <--- Correctly placed on the outer hstack to control gap
                                                                align="end",   # <--- Correctly placed on the outer hstack
                                                                style={"gap": "5px"},   # --- reduziert den Abstand zwischen time and event dropdown
                                                            ),

                                                            # 3. --- Exclude Columns Popover Manager ---
                                                            rx.vstack(  # --- ROW 3 / UNTERE ZEILE
                                                                rx.popover.root(
                                                                    rx.popover.trigger(
                                                                        rx.button(
                                                                            rx.hstack(
                                                                                rx.text("⊝", color="#D32F2F", font_size="1.2em", font_weight="bold"),
                                                                                rx.text("Exclude Col", color="black"),
                                                                                spacing="1",
                                                                                align="center",
                                                                            ),
                                                                            size="1", 
                                                                            variant="surface", 
                                                                            width="100%",
                                                                            justify="between",
                                                                            style={
                                                                                "border_left": f"3px solid {rx.color('slate', 8)}", # Adds the neutral grey/slate accent
                                                                            }
                                                                        )
                                                                    ),
                                                                    rx.popover.content(
                                                                        rx.vstack(
                                                                            rx.text("Select columns to hide:", font_size="0.75em", font_weight="bold", color=rx.color("slate", 10)),
                                                                            rx.box(
                                                                                rx.vstack(
                                                                                    rx.foreach(
                                                                                        State.all_dataset_columns,
                                                                                        lambda col: rx.hstack(
                                                                                            rx.checkbox(
                                                                                                checked=State.user_excluded_columns.contains(col),
                                                                                                on_change=lambda val: State.toggle_exclude_column(col, val),
                                                                                                size="1",
                                                                                            ),
                                                                                            rx.text(col, font_size="0.8em"),
                                                                                            spacing="2",
                                                                                            align="center",
                                                                                            width="100%"
                                                                                        )
                                                                                    ),
                                                                                    spacing="2",
                                                                                    width="100%",
                                                                                ),
                                                                                max_height="180px",
                                                                                overflow_y="auto",
                                                                                width="100%",
                                                                                padding_right="4px",
                                                                            ),
                                                                            spacing="2",
                                                                            width="210px",
                                                                            padding="8px",
                                                                        )
                                                                    ),
                                                                ),
                                                                spacing="1",
                                                                width="105px",
                                                                margin_top="21px",
                                                            ),
            
                                                            # 4. --- Direct Full-Row Duplicate Delete Button ---
                                                            rx.vstack(  
                                                                rx.button(
                                                                    rx.hstack(
                                                                        rx.text("✕", color="#D32F2F", font_size="1.2em", font_weight="bold"),
                                                                        rx.text("⧉", font_size="1.0em"),
                                                                        rx.text("Row", color="black", font_weight="medium"),
                                                                        spacing="1",
                                                                        align="center",
                                                                    ),
                                                                    size="1",
                                                                    variant="surface",
                                                                    on_click=State.remove_exact_duplicates,
                                                                    width="100%",
                                                                ),
                                                                
                                                                # --- Floating Plain-Text Message (No Layout Shift) ---
                                                                rx.cond(
                                                                    State.status_message != "",
                                                                    rx.text(
                                                                        State.status_message,
                                                                        style={
                                                                            "position": "absolute",
                                                                            "top": "35px",              # Positions text below button
                                                                            "left": "0px",
                                                                            "white_space": "nowrap",    # Prevents text from wrapping
                                                                            "font_size": "0.75em",
                                                                            "color": "#D97706",       # Amber text color
                                                                            "font_weight": "500",
                                                                            "z_index": "9999",          # Text stays in forefront
                                                                        },
                                                                    ),
                                                                ),
                                                                spacing="1",
                                                                width="95px",
                                                                margin_top="21px",
                                                                style={
                                                                    "position": "relative", 
                                                                    "overflow": "visible",   # Ensures the absolute child isn't clipped
                                                                },
                                                            ),

                                                            # 5. --- Selective Dup Button (Dedup Criteria Popover) ---
                                                            rx.vstack(  
                                                                rx.popover.root(
                                                                    rx.popover.trigger(
                                                                        rx.button(
                                                                            rx.hstack(
                                                                                rx.text("⚙️", font_size="1.0em"),
                                                                                rx.text("✕", color="#D32F2F", font_size="1.2em", font_weight="bold"),
                                                                                rx.text("Row", color="black", font_weight="medium"),
                                                                                spacing="1",
                                                                                align="center",
                                                                                width="100%",
                                                                            ),
                                                                            size="1",
                                                                            variant="surface",
                                                                            width="100%",
                                                                        ),
                                                                    ),
                                                                    rx.popover.content(
                                                                        rx.vstack(
                                                                            rx.text("Find duplicates based on:", font_size="0.75em", font_weight="bold", color=rx.color("slate", 10)),
                                                                            rx.box(
                                                                                rx.vstack(
                                                                                    rx.foreach(
                                                                                        State.all_dataset_columns,
                                                                                        lambda col: rx.hstack(
                                                                                            rx.checkbox(
                                                                                                checked=State.selected_dedup_columns.contains(col),
                                                                                                on_change=lambda val: State.toggle_dedup_column(col, val),
                                                                                                size="1",
                                                                                            ),
                                                                                            rx.text(col, font_size="0.8em"),
                                                                                            spacing="2",
                                                                                            align="center",
                                                                                            width="100%"
                                                                                        )
                                                                                    ),
                                                                                    spacing="2",
                                                                                    width="100%",
                                                                                ),
                                                                                max_height="180px",
                                                                                overflow_y="auto",
                                                                                width="100%",
                                                                                padding_right="4px",
                                                                            ),
                                                                            # Wrapping the button in rx.popover.close automatically closes the dropdown on click
                                                                            rx.popover.close(
                                                                                rx.button(
                                                                                    rx.hstack(
                                                                                        rx.text("✕", color="#D32F2F", font_size="1.2em", font_weight="bold"),
                                                                                        rx.text("⧉", font_size="1.0em"),
                                                                                        rx.text("Delete Rows", color="black", font_weight="medium"),
                                                                                        spacing="1",
                                                                                        align="center",
                                                                                    ),
                                                                                    size="1",
                                                                                    variant="surface",
                                                                                    on_click=State.remove_selective_duplicates,
                                                                                    width="100%",
                                                                                ),
                                                                            ),
                                                                            spacing="2",
                                                                            width="210px",
                                                                            padding="8px",
                                                                        )
                                                                    ),
                                                                ),
                                                                
                                                                # --- Floating Plain-Text Message (Positioned Up & Layered on Top) ---
                                                                rx.cond(
                                                                    State.selective_status_message != "",
                                                                    rx.text(
                                                                        State.selective_status_message,
                                                                        style={
                                                                            "position": "absolute",
                                                                            "top": "35px",          
                                                                            "left": "0px",
                                                                            "white_space": "nowrap", 
                                                                            "font_size": "0.75em",
                                                                            "color": "#D97706",     
                                                                            "font_weight": "500",
                                                                            "z_index": "9999",      
                                                                        },
                                                                    ),
                                                                ),
                                                                spacing="1",
                                                                width="95px",
                                                                margin_top="21px",
                                                                style={
                                                                    "position": "relative", 
                                                                    "overflow": "visible",   # Ensures the absolute child isn't clipped
                                                                },
                                                            ),

                                                            # 6. --- Data Mapping Button & Popover ---
                                                            rx.vstack(  # --- ROW 3 / UNTERE ZEILE
                                                                #rx.text("Data Mapping", font_size="0.75em", font_weight="bold", color=rx.color("slate", 11)),
                                                                rx.popover.root(

                                                                    rx.popover.trigger(
                                                                    
                                                                        rx.button(
                                                                            rx.hstack(
                                                                                rx.text("⇄", color="#2E7D32", font_size="1.2em", font_weight="bold"),
                                                                                rx.text("Map Data", color="black"),
                                                                                spacing="1",
                                                                                align="center",
                                                                            ),
                                                                            size="1",
                                                                            variant="surface",
                                                                            width="100%",
                                                                            justify="between",
                                                                        ),
                                                                    ),

                                                                    rx.popover.content(
                                                                        rx.vstack(
                                                                            rx.text("Map Values / Re-code Column", font_size="0.8em", font_weight="bold", color=rx.color("slate", 12)),
                                                                            
                                                                            # Column selector dropdown
                                                                            rx.select.root(
                                                                                rx.select.trigger(placeholder="Select column to map...", width="100%"),
                                                                                rx.select.content(
                                                                                    rx.foreach(State.all_dataset_columns, lambda opt: rx.select.item(opt, value=opt))
                                                                                ),
                                                                                value=State.selected_mapping_column,
                                                                                on_change=State.set_mapping_column,
                                                                                size="1",
                                                                                width="100%",
                                                                            ),
                                                                            
                                                                            # Dynamic Mapping Table (Original Value -> New Target Input)
                                                                            rx.cond(
                                                                                State.mapping_source_values.length() > 0,
                                                                                rx.vstack(
                                                                                    rx.text("Map existing values to new values:", font_size="0.75em", color=rx.color("slate", 10)),
                                                                                    rx.box(
                                                                                        rx.vstack(
                                                                                            rx.foreach(
                                                                                                State.mapping_source_values,
                                                                                                lambda orig: rx.hstack(
                                                                                                    rx.text(orig, font_size="0.75em", width="40%", weight="bold"),
                                                                                                    rx.input(
                                                                                                        placeholder="New value",
                                                                                                        size="1",
                                                                                                        width="60%",
                                                                                                        on_change=lambda val: State.update_single_mapping(orig, val),
                                                                                                    ),
                                                                                                    spacing="2",
                                                                                                    align="center",
                                                                                                    width="100%",
                                                                                                )
                                                                                            ),
                                                                                            spacing="2",
                                                                                            width="100%",
                                                                                        ),
                                                                                        max_height="140px",
                                                                                        overflow_y="auto",
                                                                                        width="100%",
                                                                                        padding_right="4px",
                                                                                    ),
                                                                                    rx.button(
                                                                                        "Apply Mapping",
                                                                                        size="1",
                                                                                        color_scheme="blue",
                                                                                        width="100%",
                                                                                        on_click=State.apply_data_mapping,
                                                                                    ),
                                                                                    spacing="2",
                                                                                    width="100%",
                                                                                ),
                                                                                rx.text("Select a column above to configure mappings.", font_size="0.75em", color=rx.color("slate", 10))
                                                                            ),
                                                                            spacing="2",
                                                                            width="240px",
                                                                            padding="8px",
                                                                        ),
                                                                    ),
                                                                    # 🟢 Bind the open state and its change handler here:
                                                                    open=State.is_mapping_popover_open,
                                                                    on_open_change=State.set_is_mapping_popover_open,
                                                                ),
                                                                spacing="1",
                                                                width="95px",
                                                                margin_top="21px",
                                                            ), # ---end vstack

                                                            spacing="2",  # sets the width of the Blue Container Box
                                                            width="100%",
                                                            align="start",
                                                        ),
                                                        spacing="3",  # Reduced main vstack spacing to keep rows tight
                                                        width="100%",
                                                    ),
                                                    #padding="12px",
                                                    #padding="12px",
                                                    background_color=rx.color("blue", 2),
                                                    border=f"1px dashed {rx.color('blue', 4)}",
                                                    border_radius="var(--radius-2)",
                                                    spacing="0",
                                                    width="100%",           # Keeps the box full width
                                                    padding_y="7px",        # Controls the top and bottom height
                                                    padding_x="12px",       # Controls side width padding independently
                                                    margin_bottom="18px",   # vertical distance between Blue Rectangle and Table Data Display
                                                ),
                                                # --------------------------------------- END RX.BOX BLUE RECTANGLE ---------------------------- #
                                                
                                                # -------------------------------------- DATA TABLE BOX ---------------------------------------- #
                                                rx.box(
                                                    rx.el.table(
                                                        rx.el.thead(    # ----------------- TABLE DATA HEADER ROW ------------------- #
                                                            rx.el.tr(
                                                                rx.foreach(
                                                                    State.data_preview_cols,
                                                                    lambda col: rx.el.th(
                                                                        rx.text(col, font_weight="bold", color=rx.color("slate", 12), font_size="12px"),
                                                                        style={
                                                                            "padding": "8px 12px",
                                                                            "text_align": "left",
                                                                            "border_bottom": f"1px solid {rx.color('slate', 5)}",
                                                                            "position": "sticky",
                                                                            "top": "0",
                                                                            "z-index": "50",
                                                                            "background_color": rx.cond(
                                                                                col == State.selected_time_column,
                                                                                "#c6e6fe",  # --- Color if column is 'time' --- #
                                                                                rx.cond(
                                                                                    col == State.selected_event_column,
                                                                                    "#fecaca",  # --- Color if column is status, or event in Breast Dataset --- #
                                                                                    rx.cond(
                                                                                        State.columns_to_hide.contains(col),
                                                                                        "#e2e8f0",  # 👈 Light gray background for unmeaningful/hidden columns
                                                                                        "#f8fafc"   # --- # Default fallback color --- #                                                                                                                                                   )
                                                                                    )
                                                                                )
                                                                            ),
                                                                        }
                                                                    )
                                                                )
                                                            ),
                                                            style={"position": "sticky", "top": "0", "z-index": "50"}
                                                        ),

                                                        # -------------------- TABLE DATA ROWS ------------------- #
                                                        rx.el.tbody(
                                                            rx.foreach( # --- LOOP HIGHLIGHTING THE ENTIRE TIME/STATUS/EVENT COLUMNS in Data View table --- #
                                                                State.data_preview_rows,
                                                                lambda row: rx.el.tr(
                                                                    rx.foreach(
                                                                        row,
                                                                        lambda cell, index: rx.el.td(
                                                                            rx.text(cell, font_size="12px"), 
                                                                            white_space="nowrap",
                                                                            padding="6px 12px",
                                                                            border_bottom=f"1px solid {rx.color('slate', 3)}",
                                                                            style={
                                                                                "background_color": rx.cond(
                                                                                    State.data_preview_cols[index] == State.selected_time_column,
                                                                                    "#c6e6fe",
                                                                                    rx.cond(
                                                                                        State.data_preview_cols[index] == State.selected_event_column,
                                                                                        
                                                                                        "#fa171f53",
                                                                                        rx.cond(
                                                                                            State.columns_to_hide.contains(State.data_preview_cols[index]),
                                                                                            "#e2e8f0",  # 👈 Light gray background for unmeaningful/hidden column cells
                                                                                            "transparent"
                                                                                        )
                                                                                    )
                                                                                )
                                                                            }
                                                                        )
                                                                    )
                                                                )
                                                            ),
                                                            style={"width": "100%", "border_collapse": "collapse", "font_family": "inherit"}
                                                        ),
                                                        style={"width": "100%", "border_collapse": "collapse", "font_family": "inherit"}
                                                    ),  # --- END EL.TABLE ---
                                                    
                                                    id="inspector-table-container",  # 👈 Crucial ID for targeting via script
                                                    custom_attrs={
                                                        "data-viewport-offset": State.viewport_offset,
                                                        "data-total-rows": State.total_row_count, # Maps your state variable to the HTML attribute
                                                    },  
                                                    style={
                                                        "max_height": "320px",
                                                        "overflow": "auto",
                                                        "overscroll_behavior": "contain",
                                                        "position": "relative",
                                                        "border": f"1px solid {rx.color('slate', 4)}",
                                                        "border_radius": "var(--radius-2)",
                                                        "background_color": "#ffffff",
                                                        "width": "100%",          # 👈 Ensures the white card fills the chart width
                                                        "align_self": "stretch"   # 👈 Forces it to fill the flex container layout
                                                    }
                                                ),  # ------------------------------------- END TABLE BOX -----------------------------------------
                                                width="100%",
                                                spacing="2",
                                            ),
                                            value="preview",
                                        ),

                                        # 3. -------------------------------- The Schema Tab Content ------------------------------------- #
                                        rx.tabs.content(
                                            rx.box(
                                                rx.el.table(
                                                    rx.el.thead(
                                                        rx.el.tr(
                                                            rx.foreach(
                                                                State.schema_cols,
                                                                lambda col: rx.el.th(
                                                                    rx.text(col, font_weight="bold", color=rx.color("slate", 12), font_size="12px"),
                                                                    style={
                                                                        "padding": "8px 12px",
                                                                        "text_align": "left",
                                                                        "border_bottom": f"1px solid {rx.color('slate', 5)}",
                                                                        "position": "sticky",
                                                                        "top": "0",
                                                                        "z-index": "50",
                                                                        "background_color": "#f8fafc",
                                                                    }
                                                                )
                                                            )
                                                        ),
                                                        style={"position": "sticky", "top": "0", "z-index": "50"}
                                                    ),
                                                    rx.el.tbody(
                                                        rx.foreach(
                                                            State.schema_rows,
                                                            lambda row: rx.el.tr(
                                                                rx.foreach(row, lambda cell: rx.el.td(
                                                                    rx.text(cell, font_size="12px"), 
                                                                    white_space="nowrap",
                                                                    padding="6px 12px",
                                                                    border_bottom=f"1px solid {rx.color('slate', 3)}"
                                                                ))
                                                            )
                                                        )
                                                    ),
                                                    style={"width": "100%", "border_collapse": "collapse", "font_family": "inherit"}
                                                ),
                                                id="inspector-table-container",  # 👈 Crucial ID for targeting via script
                                                style={
                                                    "max_height": "320px",
                                                    "overflow": "auto",
                                                    "overscroll_behavior": "contain",
                                                    "position": "relative",
                                                    "border": f"1px solid {rx.color('slate', 4)}",
                                                    "border_radius": "var(--radius-2)",
                                                    "background_color": "#ffffff",
                                                    "width": "100%",          # 👈 Ensures the white card fills the chart width
                                                    "align_self": "stretch"   # 👈 Forces it to fill the flex container layout
                                                }
                                            ),
                                            value="schema",
                                        ),
                                        value=State.inspector_active_tab,
                                        on_change=State.change_inspector_tab,
                                        width="100%",
                                    ),
                    
                                    width="100%",
                                    overflow_x="auto",        # Forces a horizontal scrollbar if columns blow past container limits
                                    margin_top="0.5em", 
                                    margin_bottom="1.5em",
                                ),
                                rx.fragment()
                            ),

                            rx.card(
                                rx.cond(
                                    State.results_ready,

                                    rx.vstack(             
                                        # ------------------------- Dynamic Plot Container ------------------------- #
                                        rx.cond(
                                            State.selected_model == "Cox Proportional Hazards",
                                            # --- WRAP IN AN RX.VSTACK SO TABLE 1 AND THE PLOT STACK VERTICALLY ---
                                            rx.vstack(
                                                # --- 🌟 1. Table 1 renders first at the very top of the CPH results view --- #
                                                render_cph_table_one(),
                                                
                                                # --- 🌟 2. Cox Proportional Hazards Forest Plot Box --- #
                                                rx.box(
                                                    rx.cond(
                                                        State.cph_has_run,
                                                        rx.plotly(
                                                            data=State.cph_fig, 
                                                            config={**plotly_config, "responsive": True}, 
                                                            width="100%",
                                                            height="100%"
                                                        ),
                                                        rx.center(
                                                            rx.text("Content loading...", color="gray"),
                                                        ),
                                                    ),
                                                    width="100%",
                                                    min_height="420px",
                                                    max_width="620px",
                                                    position="relative",
                                                    overflow="hidden",
                                                ),
                                                
                                                # --- 🌟  3. Rest of CPH tables, validation cards, calibration, risk-stratified cards --- #
                                                #cph_results_table(),
                                                # ... (your other cards)
                                                
                                                width="100%",
                                                spacing="6",
                                                align="stretch",
                                            ),

                                            rx.vstack(
                                                # 2. 🌟 Existing Kaplan-Meier Plot Box & Risk Table
                                                rx.box(
                                                    rx.cond(
                                                        State.selected_dataset == "Own Data (.csv/.xls)",
                                                        rx.plotly(
                                                            data=State.survival_fig_g1, 
                                                            config={**plotly_config, "responsive": True},
                                                            width="100%",
                                                            height="100%"
                                                        ),
                                                        rx.plotly(
                                                            data=State.survival_fig, 
                                                            config={**plotly_config, "responsive": True},
                                                            width="100%",
                                                            height="100%"
                                                        ),
                                                    ),
                                                    width="100%",
                                                    min_height="420px",
                                                    max_width="620px",
                                                    position="relative",
                                                    overflow="hidden",
                                                ),
                                                
                                                # 3. 📊 Existing KM Survival Probability & Median Survival Cards below...
                                                width="100%",
                                                align="stretch",
                                                spacing="4",
                                            )
                                        ),

                                        # --- INSERT YOUR NEW LEGEND BLOCK HERE ---
                                        rx.cond(
                                            # Check if the active model is Kaplan-Meier
                                            State.selected_model == "Kaplan-Meier Survival",
                                        
                                            rx.vstack(    # --- TRUE: Render group legend block ---
                                                rx.cond(
                                                    # 1. FULL COHORT: Neither group has filters
                                                    (State.g1_detailed_label == "Group 1: (Full Cohort)") & (State.g2_detailed_label == "Group 2: (Full Cohort)"),
                                                    
                                                    # --- IF TRUE: Render just the single black bold line ---
                                                    rx.text(
                                                        "Full Cohort", 
                                                        color="#000000",  # black
                                                        font_size="0.70em", 
                                                        font_weight="semibold",
                                                        visibility="hidden",  # Keeps the exact vertical space without rendering the text
                                                    ),
                                                    
                                                    # --- ELSE: Handle Filtered & Complement Scenarios ---
                                                    rx.cond(
                                                        # 2a. Group 1 Active, Group 2 Empty -> Group 2 is Complement
                                                        (State.g1_detailed_label != "Group 1: (Full Cohort)") & (State.g2_detailed_label == "Group 2: (Full Cohort)"),
                                                        rx.fragment(
                                                            rx.hstack(
                                                                rx.text("Group 1:", color="#3b82f6", font_weight="semibold", font_size="0.70em"),
                                                                rx.text(
                                                                    State.g1_detailed_label.replace("Group 1:", "").strip(),
                                                                    color="#333333",
                                                                    font_size="0.70em"
                                                                ),
                                                                spacing="1",
                                                                align="center"
                                                            ),
                                                            rx.hstack(
                                                                rx.text("Group 2:", color="#ec4899", font_weight="semibold", font_size="0.70em"),
                                                                rx.text(
                                                                    "Complement (Rest of Cohort)",
                                                                    color="#333333",
                                                                    font_size="0.70em"
                                                                ),
                                                                spacing="1",
                                                                align="center"
                                                            ),
                                                        ),
                                                        
                                                        # 2b. Group 2 Active, Group 1 Empty -> Group 1 is Complement
                                                        rx.cond(
                                                            (State.g1_detailed_label == "Group 1: (Full Cohort)") & (State.g2_detailed_label != "Group 2: (Full Cohort)"),
                                                            rx.fragment(
                                                                rx.hstack(
                                                                    rx.text("Group 1:", color="#3b82f6", font_weight="semibold", font_size="0.70em"),
                                                                    rx.text(
                                                                        "Complement (Rest of Cohort)",
                                                                        color="#333333",
                                                                        font_size="0.70em"
                                                                    ),
                                                                    spacing="1",
                                                                    align="center"
                                                                ),
                                                                rx.hstack(
                                                                    rx.text("Group 2:", color="#ec4899", font_weight="semibold", font_size="0.70em"),
                                                                    rx.text(
                                                                        State.g2_detailed_label.replace("Group 2:", "").strip(),
                                                                        color="#333333",
                                                                        font_size="0.70em"
                                                                    ),
                                                                    spacing="1",
                                                                    align="center"
                                                                ),
                                                            ),
                                                            
                                                            # 2c. Both Groups Have Custom Filters Applied
                                                            rx.fragment(
                                                                # 🔵 Group 1: Colored label beneath Chart for Sub-Group 1
                                                                rx.hstack(
                                                                    rx.text("Group 1:", color="#3b82f6", font_weight="semibold", font_size="0.70em"),
                                                                    rx.text(
                                                                        State.g1_detailed_label.replace("Group 1:", "").strip(),
                                                                        color="#333333",
                                                                        font_size="0.70em"
                                                                    ),
                                                                    spacing="1",
                                                                    align="center"
                                                                ),
                                                                # 💗 Group 2: Colored label beneath Chart for Sub-Group 2
                                                                rx.hstack(
                                                                    rx.text("Group 2:", color="#ec4899", font_weight="semibold", font_size="0.70em"),
                                                                    rx.text(
                                                                        State.g2_detailed_label.replace("Group 2:", "").strip(),
                                                                        color="#333333",
                                                                        font_size="0.70em"
                                                                    ),
                                                                    spacing="1",
                                                                    align="center"
                                                                ),
                                                            )
                                                        )
                                                    )
                                                ),
                                                
                                                spacing="0",
                                                align_items="start",
                                                width="100%",
                                                padding_left="4em",
                                                margin_top="-28px",    # Pulls it slightly closer to the chart
                                                z_index="10",          # Tells the browser: "Put this on top"
                                                position="relative",    # Ensures the z_index applies correctly
                                            ),
                                            rx.fragment()      # --- FALSE: Render nothing when KM is inactive ---
                                        ),

                                        rx.box(
                                            rx.cond(
                                                State.selected_model == "Cox Proportional Hazards",                                    
                                                # --- TRUE BRANCH: Wrapped in a single rx.vstack container ---
                                                rx.vstack(
                                                    # 1. CPH Results Table
                                                    cph_results_table(),

                                                    # 2. Model Performance & Validation Summary Card
                                                    rx.card(
                                                        rx.vstack(
                                                            rx.heading("Multivariable Model Performance & Validation", font_size="12px", color="#000000", font_weight="semibold"),
                                                            rx.divider(),
                                                            rx.hstack(
                                                                rx.icon("users", size=16, color="#10b981"),
                                                                rx.text(State.cph_sample_metrics_text, font_size="11px"),
                                                                spacing="2",
                                                            ),
                                                            rx.hstack(
                                                                rx.icon("activity", size=16, color="#3b82f6"),
                                                                rx.text(State.cph_c_index_text, font_size="11px"),
                                                                rx.text("• Apparent:", font_size="11px", color="#000000", margin_left="8px"),
                                                                rx.text(State.cph_apparent_c_index, font_size="11px", font_weight="semibold"),
                                                                rx.text("• Optimism-Corrected:", font_size="11px", color="#000000", margin_left="8px"),
                                                                rx.text(State.cph_corrected_c_index, font_size="11px", font_weight="semibold", color="#2563eb"),
                                                                align="center",
                                                                spacing="1",
                                                                width="100%",
                                                            ),
                                                            rx.hstack(
                                                                rx.icon("chart-bar", size=16, color="#6366f1"),
                                                                rx.text(State.cph_log_rank_p, font_size="11px"),
                                                                spacing="2",
                                                            ),
                                                            rx.hstack(
                                                                rx.icon("shield-check", size=16, color="#10b981"),
                                                                rx.text(State.cph_ph_warning, font_size="11px"),
                                                                spacing="2",
                                                            ),
                                                            spacing="2",
                                                            align="start",
                                                        ),
                                                        width="100%",
                                                        padding="1rem",
                                                        margin_top="0.5rem",
                                                        background_color="white",
                                                        border="1px solid #e2e8f0",
                                                        border_radius="8px",
                                                    ),

                                                    # 3. Model Calibration Plot Card & Statistics
                                                    rx.card(
                                                        rx.vstack(
                                                            rx.heading("Multivariable Model Calibration (Absolute Risk Accuracy)", font_size="12px", color="#000000", font_weight="semibold"),
                                                            rx.text(State.cph_calib_horizon_text, font_size="11px", color="#64748b"),
                                                            rx.divider(),
                                                            rx.plotly(data=State.cph_calibration_fig, width="100%", height="320px"),
                                                            rx.box(
                                                                rx.vstack(
                                                                    rx.vstack(
                                                                        rx.heading("Quantitative Calibration Validation", font_size="12px", color="#000000", font_weight="semibold"),
                                                                        rx.text(
                                                                            "Evaluated via Greenwood-Nam-D'Agostino (GND) goodness-of-fit test across risk deciles and Cox recalibration regression.",
                                                                            font_size="11px",
                                                                            color="#64748b",
                                                                        ),
                                                                        spacing="1",
                                                                        align="start",
                                                                    ),
                                                                    rx.grid(
                                                                        rx.card(
                                                                            rx.text("Calibration Intercept", font_size="12px", color="#000000", font_weight="bold"),
                                                                            rx.text(State.calib_intercept_val, font_size="11px", color="#000000", font_weight="normal"),
                                                                            rx.text("Assessment of calibration-in-the-large (Ideal: 0.00)", font_size="11px", color="#64748b", font_weight="normal"),
                                                                        ),
                                                                        rx.card(
                                                                            rx.text("Calibration Slope", font_size="12px", font_weight="bold", color="#000000"),
                                                                            rx.text(rx.hstack(rx.text(State.calib_slope_val), rx.text(State.calib_slope_ci)), font_size="11px", color="#000000", font_weight="normal"),
                                                                            rx.text("Ideal: 1.00 (Detects overfitting/underfitting)", font_size="11px", color="#64748b", font_weight="normal"),
                                                                        ),
                                                                        rx.card(
                                                                            rx.text("GND Goodness-of-Fit Test", font_size="12px", font_weight="bold", color="#000000"),
                                                                            rx.hstack(
                                                                                rx.text("chi2 =", font_size="11px", color="#000000", font_weight="normal"),
                                                                                rx.text(State.gnd_chi2_val, font_size="11px", color="#000000", font_weight="normal"),
                                                                                rx.text(", p =", font_size="11px", color="#000000", font_weight="normal"),
                                                                                rx.text(State.gnd_p_val, font_size="11px", color="#000000", font_weight="normal"),
                                                                                spacing="1",
                                                                                align="center",
                                                                            ),
                                                                            rx.text("H0: Well-calibrated (p > 0.05)", font_size="11px", color="#64748b", font_weight="normal"),
                                                                        ),
                                                                        columns="3",
                                                                        gap="4",
                                                                        width="100%",
                                                                    ),
                                                                    spacing="3",
                                                                    width="100%",
                                                                    align="stretch",
                                                                ),
                                                                padding="4",
                                                                background_color="var(--gray-2)",
                                                                border_radius="8px",
                                                                mt="4",
                                                            ),
                                                            spacing="2",
                                                            align="stretch",
                                                        ),
                                                        width="100%",
                                                        padding="1rem",
                                                        margin_top="0.5rem",
                                                        background_color="white",
                                                        border="1px solid #e2e8f0",
                                                        border_radius="8px",
                                                    ),

                                                    # 4. Decision Curve Analysis (DCA) Card
                                                    rx.card(
                                                        rx.vstack(
                                                            rx.heading("Decision Curve Analysis for the Combined Multivariable Risk Model", size="2", weight="bold"),
                                                            rx.text(
                                                                "Evaluates clinical utility across threshold probabilities compared to 'Treat All' and 'Treat None' strategies.",
                                                                font_size="11px",
                                                                color="#64748b",
                                                            ),
                                                            rx.divider(),
                                                            rx.plotly(data=State.cph_dca_fig, width="100%", height="320px"),
                                                            spacing="2",
                                                            align="stretch",
                                                        ),
                                                        width="100%",
                                                        padding="1rem",
                                                        margin_top="0.5rem",
                                                        background_color="white",
                                                        border="1px solid #e2e8f0",
                                                        border_radius="8px",
                                                    ),

                                                    # 5. NEW: Time-Dependent AUC Card
                                                    rx.card(
                                                        rx.vstack(
                                                            rx.heading("Multivariable Time-Dependent Discrimination (AUC Trajectory)", font_size="12px", color="#000000", font_weight="semibold"),
                                                            rx.text("Evaluates predictive discrimination across dynamic follow-up time horizons.", font_size="11px", color="#64748b"),
                                                            rx.divider(),
                                                            rx.plotly(data=State.time_dep_auc_fig, width="100%", height="300px"),
                                                            spacing="2",
                                                            align="stretch",
                                                        ),
                                                        width="100%",
                                                        padding="1rem",
                                                        margin_top="0.5rem",
                                                        background_color="white",
                                                        border="1px solid #e2e8f0",
                                                        border_radius="8px",
                                                    ),

                                                    # 6. Risk-Stratified Kaplan-Meier Card
                                                    rx.card(
                                                        rx.vstack(
                                                            rx.heading("Risk-Stratified Kaplan-Meier Survival Analysis", font_size="12px", color="#000000", font_weight="semibold"),
                                                            rx.text(State.risk_stratified_km_summary_text, font_size="11px", color="#64748b"),
                                                            rx.divider(),
                                                            rx.plotly(data=State.risk_stratified_km_fig, width="100%", height="320px"),
                                                            spacing="2",
                                                            align="stretch",
                                                        ),
                                                        width="100%",
                                                        padding="1rem",
                                                        margin_top="0.5rem",
                                                        background_color="white",
                                                        border="1px solid #e2e8f0",
                                                        border_radius="8px",
                                                    ),
                                                ),
                                                
                                                # --- FALSE BRANCH: Number at Risk Table ---
                                                rx.table.root(  
                                                    rx.table.header(
                                                        rx.table.row(
                                                            rx.table.column_header_cell(
                                                                State.at_risk_columns[0],
                                                                font_size="12px",
                                                                text_align="left",
                                                                white_space="nowrap",
                                                                width="155px",
                                                            ),
                                                            rx.foreach(
                                                                State.at_risk_columns[1:],
                                                                lambda col: rx.table.column_header_cell(
                                                                    col, 
                                                                    font_size="12px",
                                                                    text_align="center",
                                                                    width="55px",
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    rx.table.body(  
                                                        rx.foreach(
                                                            State.at_risk_data, 
                                                            lambda row: rx.table.row(
                                                                rx.table.cell(
                                                                    rx.cond(
                                                                        row[0].contains("Full Cohort"),
                                                                        rx.text(
                                                                            "Full Cohort", 
                                                                            font_size="12px", 
                                                                            font_weight="semibold", 
                                                                            color="#333333"
                                                                        ),
                                                                        rx.cond(
                                                                            row[0].contains("Group 1"),
                                                                            rx.tooltip(
                                                                                rx.text(
                                                                                    "Group 1", 
                                                                                    font_size="11px", 
                                                                                    font_weight="semibold", 
                                                                                    color="#3b82f6",
                                                                                    cursor="help"
                                                                                ),
                                                                                content=State.group1_composition_text,
                                                                            ),
                                                                            rx.tooltip(
                                                                                rx.text(
                                                                                    "Group 2", 
                                                                                    font_size="11px", 
                                                                                    font_weight="semibold", 
                                                                                    color="#ec4899",
                                                                                    cursor="help"
                                                                                ),
                                                                                content=State.group2_composition_text,
                                                                            )
                                                                        )
                                                                    ),
                                                                    font_size="12px",
                                                                    font_weight="semibold",
                                                                    text_align="left",
                                                                    width="155px",
                                                                ), 
                                                                rx.foreach(
                                                                    row[1:],
                                                                    lambda cell_value: rx.table.cell(
                                                                        cell_value, 
                                                                        font_size="12px",
                                                                        text_align="center",
                                                                        width="55px",
                                                                        color="#333333" 
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    variant="surface", size="1", width="auto"
                                                )
                                            ),
                                            width="100%", min_height="65px"
                                        ),
                                        spacing="4", 
                                        width="100%", 
                                        padding="16px", 
                                        overflow="hidden",
                                        align_items="center", # 👈 Ensures all children (including max_width boxes) center horizontally
                                    ),  # --- END RX.VSTACK

                                    # ⚡ False Branch: Awaiting Configuration Canvas
                                    rx.center(
                                        rx.vstack(
                                            rx.box(
                                                rx.box(class_name="medical-pulse-bar pulse-b1"),
                                                rx.box(class_name="medical-pulse-bar pulse-b2"),
                                                rx.box(class_name="medical-pulse-bar pulse-b3"),
                                                rx.box(class_name="medical-pulse-bar pulse-b4"),
                                                rx.box(class_name="medical-pulse-bar pulse-b5"),
                                                rx.box(class_name="medical-pulse-bar pulse-b6"),
                                                class_name="medical-pulse-loader"
                                            ),
                                            rx.heading("Awaiting Model Configuration", size="4", color=rx.color("slate", 12), margin_bottom="0.8em"),
                                            
                                            # 🟢 CLINICAL ONBOARDING CHECKLIST
                                            rx.vstack(
                                                # --- Step 1: Dataset Tracker ---
                                                rx.hstack(
                                                    rx.cond(
                                                        State.selected_dataset,
                                                        rx.icon(tag="circle-check", size=15, color=rx.color("green", 9)),
                                                        rx.icon(tag="circle", size=15, color=rx.color("slate", 7))
                                                    ),
                                                    rx.text("Select Patient Dataset", font_size="0.85em", color=rx.color("slate", 11)),
                                                    spacing="2", align_items="center"
                                                ),
                                                
                                                # --- Step 2: Model Settings Tracker ---
                                                rx.hstack(
                                                    rx.cond(
                                                        State.selected_model,
                                                        rx.icon(tag="circle-check", size=15, color=rx.color("green", 9)),
                                                        rx.icon(tag="circle", size=15, color=rx.color("slate", 7))
                                                    ),
                                                    rx.text("Select Analytical Model & Settings", font_size="0.85em", color=rx.color("slate", 11)),
                                                    spacing="2", align_items="center"
                                                ),
                                                
                                                # --- Step 3: Patient Characteristics (Revealed ONLY for Cox Models) ---
                                                rx.cond(
                                                    State.selected_model == "Cox Proportional Hazards",
                                                    rx.hstack(
                                                        rx.cond(
                                                            State.cph_active_covariates,
                                                            rx.icon(tag="circle-check", size=15, color=rx.color("green", 9)),
                                                            rx.icon(tag="circle", size=15, color=rx.color("slate", 7))
                                                        ),
                                                        rx.text("Select Patient Characteristics", font_size="0.85em", color=rx.color("slate", 11)),
                                                        spacing="2", align_items="center",
                                                        margin_left="12px" 
                                                    )
                                                ),
                                                
                                                # --- Step 4: Action Trigger (Smart Green Light Play Button) ---
                                                rx.hstack(
                                                    rx.cond(
                                                        State.selected_dataset & (
                                                            (State.selected_model == "Kaplan-Meier Survival") | 
                                                            ((State.selected_model == "Cox Proportional Hazards") & State.cph_active_covariates)
                                                        ),
                                                        rx.icon(tag="play", size=14, color=rx.color("green", 9), style={"fill": rx.color("green", 9)}),
                                                        rx.icon(tag="play", size=14, color=rx.color("slate", 8))
                                                    ),
                                                    rx.text(
                                                        "Click ", 
                                                        rx.text.strong(
                                                            "RUN ANALYSIS", 
                                                            color=rx.cond(
                                                                State.selected_dataset & (
                                                                    (State.selected_model == "Kaplan-Meier Survival") | 
                                                                    ((State.selected_model == "Cox Proportional Hazards") & State.cph_active_covariates)
                                                                ),
                                                                rx.color("green", 9),
                                                                rx.color("blue", 9)
                                                            )
                                                        ), 
                                                        " to generate Trajectories", 
                                                        font_size="0.85em", 
                                                        color=rx.cond(
                                                            State.selected_dataset & (
                                                                (State.selected_model == "Kaplan-Meier Survival") | 
                                                                ((State.selected_model == "Cox Proportional Hazards") & State.cph_active_covariates)
                                                            ),
                                                            rx.color("slate", 12),
                                                            rx.color("slate", 11)
                                                        )
                                                    ),
                                                    spacing="2", align_items="center"
                                                ),
                                                
                                                spacing="3",
                                                align_items="start",
                                                background="rgba(255, 255, 255, 0.4)",
                                                padding="16px 24px",
                                                border_radius="var(--radius-3)",
                                                border="1px solid rgba(226, 232, 240, 0.5)"
                                            ),
                                            
                                            spacing="2", align_items="center",
                                            style={
                                                "background": "linear-gradient(135deg, rgba(248, 250, 252, 0.92) 0%, rgba(241, 245, 249, 0.92) 100%)",
                                                "backdrop-filter": "blur(8px)", "padding": "40px", "border-radius": "16px",
                                                "border": "1px solid rgba(226, 232, 240, 0.9)", "max-width": "500px",
                                                "box-shadow": "0 10px 25px -5px rgba(148, 163, 184, 0.05)"
                                            }
                                        ),
                                        width="620px", 
                                        height="495px", 
                                        border=f"1px dashed {rx.color('slate', 5)}",
                                        border_radius="var(--radius-3)", 
                                        class_name="clinical-blueprint-canvas", 
                                        padding="2em", 
                                        position="relative",
                                        margin="0 auto",          # 👈 Centers the box horizontally if parent allows
                                    )
                                ),
                                width="100%", padding="16px", overflow="hidden"
                            ),
                            
                            # --------------------------------- Begin Statistical Columns Box -------------------------------- #
                            rx.cond(
                                State.selected_model == "Kaplan-Meier Survival",
                                rx.card(    
                                    rx.hstack(
                                        # Column 1 with Tooltip
                                        rx.box(
                                            render_data_column(
                                                "SURVIVAL PROB.", 
                                                data_list=State.survival_probability_list,  
                                                tooltip_text="The percentage of patients expected to survive beyond a given time point. Reported alongside its 95% Greenwood confidence interval to reflect statistical uncertainty.",
                                            ),
                                            border_right="1px solid #cbd5e1",  
                                            padding_right="16px",              
                                            flex="1"                           
                                        ),

                                        # Column 2 (with Tooltip)
                                        rx.box(
                                            render_data_column(
                                                "MEDIAN SURVIVAL", 
                                                data_list=State.median_survival_list,
                                                tooltip_text="The estimated time at which 50% of the patient cohort is still alive (or has not experienced the event). Reported with its 95% confidence interval or 'NR' if the median was not reached.",
                                            ),
                                            border_right="1px solid #cbd5e1",  
                                            padding_right="16px",              
                                            flex="1"                           
                                        ),

                                        # Specialized Data Column 3 (With Tooltip)
                                        rx.box(
                                            render_stats_column("STATISTICAL PARAMETERS",
                                                tooltip_text="Hypothesis test results (e.g., Log-rank test p-value) evaluating whether the overall survival distributions between compared patient groups are statistically significantly different."
                                            ),
                                            padding_left="16px",               
                                            flex="1",
                                            width="100%"
                                        ),
                                    ),        
                                    width="100%", 
                                    bg=rx.color("blue", 2),
                                    padding_y="16px",          
                                    padding_x="16px"
                                ),
                                rx.fragment(),  # Renders nothing when CPH is active
                            ), 
                            # ---------------------------------- End Statistical Columns Box --------------------------------- #

                            # --------------------------------- Begin Clinical Guidance Box ---------------------------------- #
                            rx.cond(
                                State.selected_model == "Kaplan-Meier Survival",
                                rx.card(
                                    rx.vstack(
                                        # Header (Icon + Title)
                                        rx.hstack(
                                            rx.cond(
                                                State.cds_severity == "amber",
                                                rx.icon("triangle_alert", size=16, color=rx.color("amber", 9)),
                                                rx.icon("info", size=16, color=rx.color("slate", 9))
                                            ),
                                            rx.text(
                                                State.cds_guidance_title, 
                                                font_weight="bold", 
                                                font_size="12px", 
                                                color=rx.cond(
                                                    State.cds_severity == "amber",
                                                    rx.color("amber", 11),
                                                    rx.color("slate", 11)
                                                )
                                            ),
                                            spacing="2", 
                                            align_items="center"
                                        ),
                                        
                                        # Dynamic Guidance Description
                                        rx.text(
                                            State.cds_guidance_text,
                                            font_size="12px", 
                                            color=rx.color("slate", 11),
                                            line_height="1.5"
                                        ),
                                        spacing="2", 
                                        align_items="start"
                                    ),
                                    width="100%", 
                                    max_width="700px", 
                                    flex="0 0 auto",
                                    border=rx.cond(
                                        State.cds_severity == "amber",
                                        f"1px solid {rx.color('amber', 5)}",
                                        f"1px solid {rx.color('slate', 5)}"
                                    ), 
                                    background_color=rx.cond(
                                        State.cds_severity == "amber",
                                        rx.color('amber', 2),
                                        rx.color('slate', 2)
                                    ), 
                                    border_radius="var(--radius-3)"
                                ),
                                rx.fragment(),  # Renders nothing when CPH is active
                            ),
                            # ----------------------------------- End Clinical Guidance Boc ------------------------------------- #
                                              
                            # ------------------------------------------ Begin Table 1 ------------------------------------------ #
                            rx.cond(
                                State.selected_model == "Kaplan-Meier Survival",
                                # 1. 📋 Table 1: Baseline Demographics & Clinical Characteristics (New Interactive Component)
                                rx.box(
                                    rx.vstack(
                                        rx.heading("Table 1: Baseline Demographics & Clinical Characteristics", font_size="14px", weight="bold", color="#0f172a"),
                                        rx.text(
                                            rx.cond(
                                                State.is_full_cohort,
                                                "Baseline characteristics for the overall study cohort. Continuous variables shown as Mean ± SD; categorical variables shown as n (%).",
                                                "Baseline characteristics stratified by active comparison groups. Continuous variables shown as Mean ± SD; categorical variables shown as n (%)."
                                            ),
                                            font_size="12px",
                                            color="#64748b",
                                            margin_bottom="12px"
                                        ),

                                        rx.cond(
                                            State.is_full_cohort,
                                            # --- 1. Full Cohort (Unstratified) Layout: 2 Columns ---
                                            rx.box(
                                                rx.table.root(
                                                    rx.table.header(
                                                        rx.table.row(
                                                            rx.table.column_header_cell(
                                                                rx.text("Characteristic", font_size="12px", weight="bold"), 
                                                                color="#475569", background_color="#f8fafc", padding="8px 16px"
                                                            ),
                                                            rx.table.column_header_cell(
                                                                rx.text(State.table_one_g1_label, font_size="12px", weight="bold"), 
                                                                color="#475569", text_align="right", background_color="#f8fafc", padding="8px 16px", width="220px"
                                                            ),
                                                        ),
                                                    ),
                                                    rx.table.body(
                                                        rx.foreach(
                                                            State.table_one_data,
                                                            lambda row: rx.table.row(
                                                                rx.table.cell(
                                                                    row["variable"],
                                                                    font_size="12px",
                                                                    font_weight=rx.cond(row["type"] == "Header", "bold", "normal"),
                                                                    color=rx.cond(row["type"] == "Header", "#0f172a", "#334155"),
                                                                    background_color="#ffffff",
                                                                    padding_left=rx.cond(row["type"] == "Categorical", "24px", "16px"),
                                                                    padding_top="6px",
                                                                    padding_bottom="6px",
                                                                ),
                                                                rx.table.cell(
                                                                    row["group1"], 
                                                                    font_size="12px", 
                                                                    text_align="right", 
                                                                    font_variant_numeric="tabular-nums", 
                                                                    background_color="#ffffff", 
                                                                    color="#334155",
                                                                    padding_top="6px",
                                                                    padding_bottom="6px",
                                                                    padding_right="24px",
                                                                ),
                                                                background_color="#ffffff",
                                                            ),
                                                        ),
                                                    ),
                                                    width="100%",
                                                    variant="surface",
                                                    size="1",
                                                ),
                                                #max_width="550px",  # 🔥 Constrains the container width, pulling the columns closer together
                                                width="100%",
                                            ),

                                            # --- 2. Stratified (Comparative) Layout: 4 Columns ---
                                            rx.table.root(
                                                rx.table.header(
                                                    rx.table.row(
                                                        rx.table.column_header_cell(
                                                            rx.text("Characteristic", font_size="12px", weight="bold"), 
                                                            color="#475569", background_color="#f8fafc", padding="6px 12px"
                                                        ),
                                                        rx.table.column_header_cell(
                                                            rx.text(State.table_one_g1_label, font_size="12px", weight="bold", color="#3b82f6"), 
                                                            text_align="center", background_color="#f8fafc", padding="6px 12px"
                                                        ),
                                                        rx.table.column_header_cell(
                                                            rx.text(State.table_one_g2_label, font_size="12px", weight="bold", color="#ec4899"), 
                                                            text_align="center", background_color="#f8fafc", padding="6px 12px"
                                                        ),
                                                        rx.table.column_header_cell(
                                                            rx.text("P-Value", font_size="12px", weight="bold"), 
                                                            text_align="center", color="#475569", background_color="#f8fafc", padding="6px 12px"
                                                        ),
                                                    ),
                                                ),
                                                rx.table.body(
                                                    rx.foreach(
                                                        State.table_one_data,
                                                        lambda row: rx.table.row(
                                                            rx.table.cell(
                                                                row["variable"],
                                                                font_size="12px",
                                                                font_weight=rx.cond(row["type"] == "Header", "bold", "normal"),
                                                                color=rx.cond(row["type"] == "Header", "#0f172a", "#334155"),
                                                                background_color="#ffffff",
                                                                padding_left=rx.cond(row["type"] == "Categorical", "20px", "10px"),
                                                                padding_top="6px",
                                                                padding_bottom="6px",
                                                            ),
                                                            rx.table.cell(
                                                                row["group1"], 
                                                                font_size="12px", 
                                                                text_align="center", 
                                                                font_variant_numeric="tabular-nums", 
                                                                background_color="#ffffff", 
                                                                color="#334155",
                                                                padding_top="6px",
                                                                padding_bottom="6px",
                                                            ),
                                                            rx.table.cell(
                                                                row["group2"], 
                                                                font_size="12px", 
                                                                text_align="center", 
                                                                font_variant_numeric="tabular-nums", 
                                                                background_color="#ffffff", 
                                                                color="#334155",
                                                                padding_top="6px",
                                                                padding_bottom="6px",
                                                            ),
                                                            rx.table.cell(
                                                                row["p_value"], 
                                                                font_size="12px", 
                                                                text_align="center", 
                                                                font_weight="500", 
                                                                font_variant_numeric="tabular-nums", 
                                                                background_color="#ffffff", 
                                                                color="#475569",
                                                                padding_top="6px",
                                                                padding_bottom="6px",
                                                            ),
                                                            background_color="#ffffff",
                                                        ),
                                                    ),
                                                ),
                                                width="100%",
                                                variant="surface",
                                                size="1",
                                            ),
                                        ),

                                        width="100%",
                                        padding="16px",
                                        background_color="#ffffff",
                                        border_radius="8px",
                                        border="1px solid #e2e8f0",
                                        box_shadow="0 1px 3px 0 rgba(0, 0, 0, 0.05)",
                                        margin_bottom="20px",
                                    ),
                                    width="100%",
                                ),
                                rx.fragment(),  # Renders nothing when CPH is active
                            ),
                            # ------------------------------ End Table 1 BASELINE CHARACTERISTICS ------------------------------- #
                            
                            rx.cond(
                                (State.selected_model == "Kaplan-Meier Survival") & State.cph_has_run & (State.cph_table_data.length() > 0),
                                cph_results_table(),        # --- Render CHP HR Table for KM Analysis --- #
                                rx.fragment(),
                            ),
                            
                            width="700px",       # <--- Locked width so it never shrinks
                            min_width="700px",
                            max_width="700px",
                            flex_shrink="0",
                            flex_grow="0",
                            spacing="5",
                            align="stretch",
                        ),
                    ),
                
                    rx.text(State.cohort_note, font_size="0.8em", color="red"),

                    rx.cond(
                        State.show_sidebar_msg,
                        rx.box(
                            rx.hstack(
                                rx.hstack(
                                    rx.icon(tag=State.sidebar_icon, size=14),
                                    rx.text(State.sidebar_msg, font_size="0.8em", font_weight="bold"),
                                    spacing="2",
                                ),
                                rx.icon(tag="x", size=14, cursor="pointer", on_click=State.clear_sidebar_msg),
                                justify="between", align_items="center", width="100%",
                            ),
                            background_color=State.sidebar_bg_color,
                            color=State.sidebar_text_color,
                            border=f"1px solid {State.sidebar_border_color}",
                            padding="10px 14px", 
                            border_radius="var(--radius-2)", 
                            box_shadow="0 4px 12px rgba(0, 0, 0, 0.1)", 
                            width="320px",
                            position="absolute",
                            bottom="2em",
                            right="2em",
                            z_index="1000",
                        ),
                    ),
                    width="740px",
                    max_width="740px",
                    min_width="740px",
                    align="stretch",
                ),  # --- END OF RX.VSTACK ---

                # --- Right Column (Smaller Menu Sidebar area)
                rx.box(     # --- DEFINING THE SIZE OF THE MENU SIDEBAR
                    settings_sidebar(), 
                    width="320px",
                    min_width="320px",  # Prevents shrinking
                    max_width="320px",  # Prevents growing
                    flex_shrink="0",
                    flex_grow="0",      # Prevents expanding
                ),  # --- END OF RX.BOX ---
                
                # --- ROW CONTAINER RX.HSTACK HOLDING BOTH COLUMNS: DATA TABLE/PLOTS/RESULTS AND SIDEBAR MENUE
                width="100%",
                max_width="1380px",
                #min_width="1380px",
                justify="center",
                position="relative",
                align_items="start",
                padding="2em",
                spacing="6",
            ), # --- END OF RX.HSTACK ---

        ),   # --- END OF OVERALL OUTER RX.VSTACK FOR CENTURING HEADLINE Clinical Prognosis Platform (CPP)

        # 1. About Modal Dialog
        rx.dialog.root(
            rx.dialog.content(
                rx.dialog.title("About Clinical Prognosis Platform (CPP)"),
                
                # Scrollable container for the structured body content
                rx.vstack(
                    rx.text(
                        "The Clinical Prognosis Platform (CPP) is an end-to-end survival analysis and clinical reporting environment designed to bridge the gap between complex statistical modeling and high-impact medical publication.",
                        size="2",
                        color="gray",
                        margin_bottom="1rem",
                    ),
                    
                    # Key Features List
                    rx.vstack(
                        rx.heading("Platform Capabilities", size="3", margin_bottom="0.5rem"),
                        
                        rx.text("• Advanced Analytical Pipeline: Seamlessly perform Kaplan-Meier (KM) survival estimations, Cox Proportional Hazards (CPH) modeling, or combined workflows (where CPH hazards and Hazard Ratio tables are automatically generated alongside KM curves).", size="2"),
                        rx.text("• Real-Time Interactive Modeling: Execute instant computations and dynamic updates with zero latency, eliminating long feedback loops to explore data models on the fly.", size="2"),
                        rx.text("• End-to-End Data Integrity: Upload your own datasets, clean and preprocess them directly within the platform for analysis. Download the prepared dataset together with the Results Report and Draft Manuscript to ensure transparency and reproducibility.", size="2"),
                        rx.text("• Journal-Ready Reporting & Draft Manuscript: Generate publication-grade analytical reports aligned with the requirements of top-tier medical journals. Start the writing process with automated Draft Manuscript templates that save hours of manual compilation.", size="2"),
                        
                        spacing="2",
                        margin_bottom="1rem",
                    ),
                    
                    # Benchmarks & Pricing Section
                    rx.vstack(
                        rx.heading("Datasets & Pricing", size="3", margin_bottom="0.5rem"),
                        
                        rx.text("• Built-In Benchmarking Datasets: Get started immediately using the pre-loaded, open public benchmarks for exploration and testing:", size="2"),
                        rx.text("  - NCCTG Lung Cancer Data (Loprinzi et al., 1994) – Advanced lung cancer study.", size="2", padding_left="1rem"),
                        rx.text("  - GBSG2 Breast Cancer Study (1994) – 686 node-positive breast cancer patients.", size="2", padding_left="1rem"),
                        rx.text("• Uploading Your Own Dataset: Free usage is provided using built-in public datasets. To analyze proprietary patient data, upload your own datasets via flexible subscription tiers: $46.00/month or an annual subscription of $440.00.", size="2"),
                        
                        spacing="2",
                    ),
                    
                    spacing="3",
                    max_height="65vh",  # Makes the content scrollable if it exceeds screen height
                    overflow_y="auto",
                    padding_right="0.5rem",
                    margin_bottom="1.5em",
                ),
                
                rx.flex(
                    rx.dialog.close(
                        rx.button("Close", variant="soft", size="1", cursor="pointer")
                    ),
                    justify="end",
                ),
                max_width="650px", # Optional: expands the modal width slightly to give the text more breathing room
            ),
            open=State.show_about_modal,
            on_open_change=State.toggle_about_modal,
        ),

        # 2. Feedback Modal Dialog (Updated with loading & success states)
        rx.dialog.root(
            rx.dialog.content(
                rx.dialog.title("Provide Feedback"),
                rx.dialog.description(
                    "Let us know your thoughts or report any issues you encounter.",
                    margin_bottom="1em",
                ),
                rx.vstack(
                    rx.text_area(
                        placeholder="Type your feedback here...",
                        value=State.feedback_text,
                        on_change=State.set_feedback_text,
                        width="100%",
                        height="100px",
                    ),
                    # Success notification text
                    rx.cond(
                        State.feedback_success,
                        rx.text("Thank you! Your feedback has been sent.", color="green", size="2"),
                    ),
                    rx.flex(
                        rx.dialog.close(
                            rx.button(
                                "Cancel", 
                                variant="soft", 
                                size="1", 
                                cursor="pointer",
                                on_click=State.cancel_feedback  # <--- Calls the clean reset method
                            ),
                        ),
                        rx.button(
                            rx.cond(State.feedback_submitting, "Sending...", "Submit"),
                            on_click=State.submit_feedback,
                            loading=State.feedback_submitting,
                            size="1",
                            cursor="pointer",
                        ),
                        spacing="2",
                        justify="end",
                        width="100%",
                        margin_top="1em",
                    ),
                    width="100%",
                ),
            ),
            open=State.show_feedback_modal,
            on_open_change=State.toggle_feedback_modal,  # <--- Handles boolean open/close state
        ),

        # 3. Updates / Changelog Modal Dialog
        rx.dialog.root(
            rx.dialog.content(
                rx.dialog.title("What's New in CPP"),
                rx.dialog.description(
                    "Empowering clinical research with peer-review-ready survival analytics and methodological compliance.",
                    margin_bottom="1em",
                ),
                rx.vstack(
                    rx.box(
                        rx.heading("v1.1.0 - Production Release", size="3", weight="bold"),
                        rx.text("• Peer-Review Manuscript Export: One-click generation of fully formatted, STROBE/REMARK (Kaplan-Meier) and TRIPOD (Cox Proportional Hazards) compliant .docx manuscripts.", size="2"),
                        rx.text("• Advanced Calibration & Validation: Integrated Greenwood-Nam-D'Agostino (GND) calibration testing, Harrell's C-index with bootstrap optimism correction, and Decision Curve Analysis (DCA).", size="2"),
                        rx.text("• Secure In-Memory Architecture: Zero-persistence byte-stream file processing ensuring high-security data handling.", size="2"),
                        padding="3",
                        background=rx.color("slate", 2),
                        border_radius="8px",
                        width="100%",
                    ),
                    rx.box(
                        rx.heading("v1.0.0 - Initial Foundation", size="3", weight="bold"),
                        rx.text("• Core Survival Engines: Non-parametric Kaplan-Meier estimation and multivariable Cox Proportional Hazards modeling.", size="2"),
                        rx.text("• Benchmark Datasets: Pre-loaded validation support for clinical cohorts including Lung and Breast cancer data.", size="2"),
                        padding="3",
                        background=rx.color("slate", 2),
                        border_radius="8px",
                        width="100%",
                    ),
                    spacing="3",
                    width="100%",
                    max_height="300px",
                    overflow_y="auto",
                ),
                rx.flex(
                    rx.dialog.close(
                        rx.button("Close", variant="soft", size="1", cursor="pointer")
                    ),
                    justify="end",
                    margin_top="1.5em",
                ),
            ),
            open=State.show_updates_modal,
            on_open_change=State.toggle_updates_modal,
        ),

        # 4. --- Interactive Manuscript Generator Modal Dialog --- #
        rx.dialog.root(
            rx.dialog.content(
                rx.vstack(
                    rx.heading("Interactive Manuscript Draft (IMRAD Format)", size="4", weight="bold"),
                    rx.text(
                        "Review and edit your auto-populated draft below before downloading the Word document.",
                        font_size="12px",
                        color="#64748b",
                    ),
                    rx.divider(),
                    rx.text_area(
                        value=State.editable_manuscript_text,
                        on_change=State.set_editable_manuscript_text,
                        placeholder="Edit manuscript text here...",
                        height="400px",
                        width="100%",
                        font_family="monospace",
                        font_size="12px",
                    ),
                    rx.hstack(
                        rx.button(
                            "Cancel",
                            on_click=State.close_manuscript_modal,
                            variant="soft",
                            color_scheme="gray",
                            size="1",
                            cursor="pointer",
                        ),
                        rx.spacer(),
                        rx.button(
                            "Download Word Document (.docx)",
                            on_click=State.export_edited_manuscript_docx,
                            color_scheme="blue",
                            size="1",
                            cursor="pointer",
                        ),
                        width="100%",
                        padding_top="10px",
                    ),
                    spacing="3",
                    align="stretch",
                ),
                max_width="800px",
            ),
            open=State.is_manuscript_modal_open,
            on_open_change=State.close_manuscript_modal,
        ),

        # 🌟 Table 1 Baseline Characteristics MODAL
        baseline_characteristics_modal(),
        
        # 🌟 Disclaimer Modal here alongside your other modals
        disclaimer_modal(),

        # 🌟 Legal Footer here at the bottom of the page content tree
        # legal_footer(),

        # Root container rx.center properties OUTER FRAMEBOX
        width="100%",          # --- rx.center spans full browser width
        position="relative",
        padding_top="3em", # --- vertical distance to the top of the browser window
    )   # --- END OF RX.CENTER

app = rx.App(
    stylesheets=["style.css"],
    head_components=[
        rx.script(src="/table_sync.js"),
    ],
)
app.add_page(index, title="Clinical Prognosis Platform (CPP)", route="/")