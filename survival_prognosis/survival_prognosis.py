from typing import List, Union
import reflex as rx
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_lung, load_gbsg2
from lifelines.statistics import logrank_test
import logging
import asyncio

# --- Mapping Configuration ---
LUNG_MAP = {
    "Female": ("sex", 1), 
    "Male": ("sex", 2), 
    "Age Split": ("age", "split"),
    "ECOG 0": ("ph.ecog", 0),
    "ECOG 1": ("ph.ecog", 1),
    "ECOG 2": ("ph.ecog", 2),
    "ECOG 3": ("ph.ecog", 3),
}

BREAST_MAP = {
    "Hormone Therapy: Yes": ("horTh", "yes"), "Hormone Therapy: No": ("horTh", "no"),
    "Post-menopausal": ("menostat", "Post"), "Pre-menopausal": ("menostat", "Pre"),
    "Grade 1": ("tgrade", "I"), "Grade 2": ("tgrade", "II"), "Grade 3": ("tgrade", "III"),
    "Age Split": ("age", "split"), "Tumor Size Split": ("tsize", "split"),
    "Lymph Node Split": ("pnodes", "split"), "Progesterone Split": ("progrec", "split"),
    "Estrogen Split": ("estrec", "split"),
}

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

class State(rx.State):
    # Unified localized sidebar notification system
    toast_duration: int = 1500  
    sidebar_msg: str = ""
    sidebar_variant: str = "success"  
    show_sidebar_msg: bool = False
    _msg_id: int = 0  

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

    def notify_future_release(self):
        return self._show_custom_toast("Scheduled for Future Release", "info")
    
    def toggle_dialog(self):
        self.show_dialog = not self.show_dialog

    selected_dataset: str = ""  
    selected_model: str = ""

    active_controls: list[str] = rx.field(default_factory=list)
    data_selection: str = "Default"
    metadata_info: str = "" 

    # --- File Tracking & Dynamic KM States ---
    uploaded_filename: str = ""
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

    # --- New Preview Grid States ---
    data_preview_cols: list[str] = rx.field(default_factory=list)
    data_preview_rows: list[list[str]] = rx.field(default_factory=list)
    inspector_active_tab: str = "preview"

    def change_inspector_tab(self, tab_value: str):
        self.inspector_active_tab = tab_value

    def run_analysis(self):         
        self.debug_status = ""
        self.clear_sidebar_msg()
        
        if not self.selected_model or not self.selected_dataset:
            self.cohort_note = ""
            return self._show_custom_toast("Please select Model and Dataset", "error")

        if self.selected_dataset == "Own CSV File" and not self.uploaded_filename:
            return self._show_custom_toast("Please upload a CSV file first.", "error")

        self.show_dialog = False
        self.show_schema_preview = False
        return self.calculate_prognosis()
    
    def reset_state(self):
        self.selected_dataset = ""
        self.selected_model = ""
        self.results_ready = False
        self.uploaded_filename = ""
        self.uploaded_row_count = 0
        self.metadata_info = ""
        self.cohort_note = ""
        self.show_schema_preview = False
        self.schema_rows = []
        self.km_stratify_col = ""
        self.km_selected_subgroups = []
        self.km_available_subgroups = []
        self.km_numeric_split_value = ""
        self.km_is_numeric_stratify = False
        
    def update_dataset(self, value: str):
        self.selected_dataset = value
        self.active_controls = [] 
        self.cph_active_covariates = []
        self.km_stratify_col = ""
        self.km_selected_subgroups = []
        self.km_available_subgroups = []
        self.results_ready = False  
        self.survival_probability_text = ""
        self.median_survival_text = "---"
        self.p_value_text = ""
        
        self.at_risk_data = [["-", "-", "-", "-", "-", "-"]] 
        self.cph_table_data = [["-", "-", "-", "-", "-"]]
        self.ci_text = "---"
        self.cohort_note = ""

        if "Breast" in value or "GBSG2" in value:
            self.metadata_info = "Source: GBSG2 Breast Cancer Study. Created: 1994. 686 patients with node-positive cancer."
            self.show_schema_preview = False
        elif "Lung" in value or "NCCTG" in value:
            self.metadata_info = "Source: NCCTG Lung Cancer Data (Loprinzi et al.). Created: 1994. Advanced lung cancer study."
            self.show_schema_preview = False
        else:
            if self.uploaded_filename:
                self.metadata_info = f"Source: {self.uploaded_filename} ({self.uploaded_row_count} rows). Custom cohort analysis."
                self.show_schema_preview = True
            else:
                self.metadata_info = "Source: No file uploaded yet. Custom cohort analysis."
                self.show_schema_preview = False

        return self._show_custom_toast(f"{value} Selected.", "success")

    def update_model(self, value: str):
        self.selected_model = value
        self.results_ready = False  
        self.survival_probability_text = ""
        self.median_survival_text = "---"
        self.p_value_text = ""
        
        self.at_risk_data = [["-", "-", "-", "-", "-", "-"]]
        self.cph_table_data = [["-", "-", "-", "-", "-"]]
        self.cohort_note = ""

        return self._show_custom_toast(f"{value} Selected.", "success")

    slider_max_val: int = 90
    months_list: list[float] = rx.field(default_factory=lambda: [0.0, 36.0])
    uploaded_data: pd.DataFrame = pd.DataFrame()
    csv_columns: list[str] = rx.field(default_factory=list)
    is_uploading: bool = False

    age_threshold: int = 50
    size_threshold: int = 20
    node_threshold: int = 3

    survival_fig: go.Figure = rx.field(default_factory=make_initial_figure)

    at_risk_columns: list[str] = rx.field(default_factory=lambda: ["Group", "M0", "M10", "M20", "M30", "M40"])
    at_risk_data: list[list[str]] = rx.field(default_factory=lambda: [["-", "-", "-", "-", "-", "-"]])

    p_value_text: str = ""
    median_survival_text: str = "---"
    survival_probability_text: str = "Awaiting..."
    ci_text: str = "---"

    cph_active_covariates: list[str] = rx.field(default_factory=list)
    cph_fig: go.Figure = rx.field(default_factory=make_initial_figure)

    cph_survival_prob: str = "---"
    cph_log_rank_p: str = ""

    cph_table_data: list[list[str]] = rx.field(default_factory=lambda: [["-", "-", "-", "-", "-"]])
    cph_table_cols: list[str] = rx.field(default_factory=lambda: ["Variable", "HR", "95% CI Lower", "95% CI Upper", "p-value"])
    
    cohort_note: str = ""       

    def set_selected_model(self, val: str): return self.update_model(val)
    def set_months_list(self, val: list[float]): self.months_list = val
    def set_age_threshold(self, v): self.age_threshold = int(v) if v else 50
    def set_size_threshold(self, v): self.size_threshold = int(v) if v else 20
    def set_node_threshold(self, v): self.node_threshold = int(v) if v else 3

    def toggle_schema_view(self):
        self.show_schema_preview = not self.show_schema_preview

    def update_cph_covariates(self, covariate: str, checked: bool):
        self.results_ready = False
        if checked:
            if covariate not in self.cph_active_covariates: self.cph_active_covariates.append(covariate)
        else:
            if covariate in self.cph_active_covariates: self.cph_active_covariates.remove(covariate)
        self.cohort_note = ""
        return self._show_custom_toast("Covariates updated.", "success")

    def handle_km_custom_stratify_change(self, col: str):
        """Builds subcategories dynamically on continuous or categorical column rules."""
        self.km_stratify_col = col
        self.km_selected_subgroups = []
        self.results_ready = False
        
        if not col or self.uploaded_data.empty or col not in self.uploaded_data.columns:
            self.km_available_subgroups = []
            self.km_is_numeric_stratify = False
            return

        # Determine type mechanics
        series = self.uploaded_data[col]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 5:
            self.km_is_numeric_stratify = True
            median_val = float(series.median())
            self.km_numeric_split_value = f"{median_val:.2f}"
            self.km_available_subgroups = [f"<={self.km_numeric_split_value}", f">{self.km_numeric_split_value}"]
            self.km_selected_subgroups = list(self.km_available_subgroups)
        else:
            self.km_is_numeric_stratify = False
            # Clean drop missing values for discrete categorization sets
            uniques = sorted([str(x) for x in series.dropna().unique()])
            self.km_available_subgroups = uniques[:10]  # Cap bounds safely to prevent rendering crashes
            self.km_selected_subgroups = list(self.km_available_subgroups[:2]) # Select top two categories as defaults

    def update_km_custom_subgroups(self, group: str, checked: bool):
        self.results_ready = False
        if checked:
            if group not in self.km_selected_subgroups:
                self.km_selected_subgroups.append(group)
        else:
            if group in self.km_selected_subgroups:
                self.km_selected_subgroups.remove(group)

    def update_km_numeric_split(self, val: str):
        self.results_ready = False
        self.km_numeric_split_value = val
        self.km_available_subgroups = [f"<={val}", f">{val}"]
        self.km_selected_subgroups = list(self.km_available_subgroups)


    def handle_upload(self, files: list[rx.UploadFile]):
        for file in files:
            import io
            df = pd.read_csv(io.BytesIO(file.file.read()))
            self.uploaded_data = df
            self.csv_columns = df.columns.tolist()
            self.cph_active_covariates = []
            self.cohort_note = ""

            self.uploaded_filename = file.filename
            self.uploaded_row_count = len(df)

            # 1. Parse schema definitions
            self.schema_rows = [
                [str(col), str(dtype), str(df[col].notnull().sum())]
                for col, dtype in zip(df.columns, df.dtypes)
            ]

            # 2. Extract layout values for row inspection
            self.data_preview_cols = df.columns.tolist()

            # 3. Handle data format conversions
            preview_df = df.head(500).fillna("NaN")
            self.data_preview_rows = preview_df.astype(str).values.tolist()

            self.show_schema_preview = True
            self.inspector_active_tab = "preview"

            self.metadata_info = f"Source: {self.uploaded_filename} ({self.uploaded_row_count} rows). Custom cohort analysis."

            return self._show_custom_toast(f"Uploaded and processed {file.filename}.", "success")


    def handle_km_selection(self, item: str, checked: bool):
        self.results_ready = False 
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
        max_limit = 45 if "Lung" in val else 90
        if self.months_list[1] > max_limit:
            self.months_list = [0.0, float(max_limit)]
        return self.update_dataset(val)
    
    def clear_sidebar_msg(self):
        self.show_sidebar_msg = False

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
        if "Lung" in self.selected_dataset:
            return 45
        elif "Breast" in self.selected_dataset:
            return 90
        elif self.selected_dataset == "Own CSV File" and not self.uploaded_data.empty:
            time_col = next((c for c in self.uploaded_data.columns if c.lower() == 'time'), None)
            if time_col and pd.api.types.is_numeric_dtype(self.uploaded_data[time_col]):
                return int(self.uploaded_data[time_col].max() / 30.44) + 5
        return 90

    def calculate_prognosis(self):      
        if self.selected_dataset == "" or self.selected_model == "":
            return self._show_custom_toast("Please select Dataset and Model", "error")

        try:
            days_input = float(self.end_month * 30.44)

            if self.selected_model == "Kaplan-Meier Survival":
                if "Breast" in self.selected_dataset:
                    df = load_gbsg2()
                    df['event'] = df['cens'].astype(int)
                    self._run_km_engine(df, BREAST_MAP, days_input, 90, "GBSG2 Breast", [0, 12, 24, 36, 48, 60, 72, 84])
                elif "Lung" in self.selected_dataset:
                    df = load_lung().dropna(subset=['time', 'status', 'sex', 'ph.ecog'])
                    df['event'] = (df['status'] - 1).astype(int)
                    self._run_km_engine(df, LUNG_MAP, days_input, 45, "NCCTG Lung", [0, 10, 20, 30, 40])
                else: 
                    # --- Dynamic CSV Curation for KM Stratification ---
                    df = self.uploaded_data.copy()
                    time_col = next((c for c in df.columns if c.lower() == 'time'), None)
                    event_col = next((c for c in df.columns if c.lower() in ['event', 'status', 'cens']), None)
                    
                    if not time_col or not event_col:
                        raise ValueError("CSV file missing explicit 'time' and 'event' tracking attributes.")
                    
                    df['time'] = pd.to_numeric(df[time_col], errors='coerce')
                    df['event'] = pd.to_numeric(df[event_col], errors='coerce')
                    df = df.dropna(subset=['time', 'event'])
                    
                    unique_ev = df['event'].unique()
                    if not set(unique_ev).issubset({0, 1}):
                        vals = sorted(list(unique_ev))
                        if len(vals) == 2:
                            df['event'] = df['event'].map({vals[0]: 0, vals[1]: 1})
                    
                    df['event'] = df['event'].astype(int)
                    max_x_val = int(df['time'].max() / 30.44) + 1
                    
                    # Call automated stratification matrix execution
                    self._run_custom_km_engine(df, days_input, max_x_val)

            else: # Cox Proportional Hazards Execution Pipe
                if not self.cph_active_covariates:
                    return self._show_custom_toast("Select at least one Covariate", "error")

                if "Lung" in self.selected_dataset:
                    df_cph = load_lung().dropna(subset=['time', 'status'] + self.cph_active_covariates).copy()
                    df_cph['event'] = (df_cph['status'] - 1).astype(int)
                    if 'sex' in self.cph_active_covariates:
                        df_cph['sex'] = df_cph['sex'].map({1: 1, 2: 2, 'Male': 1, 'Female': 2})
                    self._run_cph_engine_isolated(df_cph, days_input, "NCCTG Lung")
                    
                elif "Breast" in self.selected_dataset:
                    df_raw = load_gbsg2().copy()
                    df_raw['event'] = df_raw['cens'].astype(int)
                    df_raw['horTh'] = df_raw['horTh'].map({'yes': 1, 'no': 0, 1: 1, 0: 0})
                    df_raw['menostat'] = df_raw['menostat'].map({'Post': 1, 'Pre': 0, 1: 1, 0: 0})
                    df_raw['tgrade'] = df_raw['tgrade'].map({'I': 1, 'II': 2, 'III': 3, 1: 1, 2: 2, 3: 3})
                    
                    keep_cols = ['time', 'event'] + self.cph_active_covariates
                    df_cph = df_raw[keep_cols].dropna().copy()
                    self._run_cph_engine_isolated(df_cph, days_input, "GBSG2 Breast")
                else: 
                    df_raw = self.uploaded_data.copy()
                    time_col = next((c for c in df_raw.columns if c.lower() == 'time'), None)
                    event_col = next((c for c in df_raw.columns if c.lower() in ['event', 'status', 'cens']), None)
                    
                    if not time_col or not event_col:
                        raise ValueError("CSV missing required temporal ('time') or event tracking columns.")
                        
                    df_raw['time'] = pd.to_numeric(df_raw[time_col], errors='coerce')
                    df_raw['event'] = pd.to_numeric(df_raw[event_col], errors='coerce')
                    
                    keep_cols = ['time', 'event'] + self.cph_active_covariates
                    df_cph = df_raw[keep_cols].copy()
                    
                    cat_cols = [c for c in self.cph_active_covariates if df_cph[c].dtype == 'object']
                    if cat_cols:
                        df_cph = pd.get_dummies(df_cph, columns=cat_cols, drop_first=True)
                        self.cph_active_covariates = [c for c in df_cph.columns if c not in ['time', 'event']]
                    
                    df_cph = df_cph.dropna()
                    self._run_cph_engine_isolated(df_cph, days_input, "Custom CSV")

            self.results_ready = True  
            self.cohort_note = ""
            return self._show_custom_toast("Analysis Complete", "success")

        except Exception as e:
            self.results_ready = False
            self.cohort_note = f"Analysis Error: {str(e)}"
            return self._show_custom_toast(f"Analysis Error: {str(e)}", "error")

    def _run_custom_km_engine(self, df, days_input, max_x):
        """Generates stratified KM survival curves dynamically based on user selections."""
        self.at_risk_data = []
        fig = go.Figure()
        
        # Calculate dynamic dynamic interval ticks based on timeline
        intervals = [0, int(max_x*0.25), int(max_x*0.5), int(max_x*0.75), int(max_x)]
        intervals = sorted(list(set(intervals)))
        interval_days = [i * 30.44 for i in intervals]
        self.at_risk_columns = ["Group"] + [f"M{i}" for i in intervals]
        
        colors = ["#3b82f6", "#ec4899", "#10b981", "#f59e0b", "#6366f1", "#8b5cf6"]
        fill_colors = [
            "rgba(59, 130, 246, 0.1)", "rgba(236, 72, 153, 0.1)", "rgba(16, 185, 129, 0.1)",
            "rgba(245, 158, 11, 0.1)", "rgba(99, 102, 241, 0.1)", "rgba(139, 92, 246, 0.1)"
        ]
        
        fitters, dataframes = [], []

        # Baseline execution if no stratification column is active
        if not self.km_stratify_col or not self.km_selected_subgroups:
            kmf = KaplanMeierFitter().fit(df['time'], df['event'], label="Baseline Cohort")
            self._add_km_trace(fig, kmf, "#475569", "rgba(71, 85, 105, 0.08)")
            fitters.append(kmf)
            dataframes.append(df)
            self.at_risk_data = [["Baseline Cohort"] + [str(self._get_at_risk(kmf, d)) for d in interval_days]]
        else:
            col = self.km_stratify_col
            if self.km_is_numeric_stratify:
                # Numeric continuous splitter split rule
                try:
                    split_v = float(self.km_numeric_split_value)
                except:
                    split_v = float(df[col].median())
                
                conds = [
                    (df[col] <= split_v, f"{col} ≤ {split_v:.1f}"),
                    (df[col] > split_v, f"{col} > {split_v:.1f}")
                ]
                for i, (cond, lbl) in enumerate(conds):
                    sub = df[cond].copy()
                    if not sub.empty:
                        kmf = KaplanMeierFitter().fit(sub['time'], sub['event'], label=lbl)
                        self._add_km_trace(fig, kmf, colors[i % 6], fill_colors[i % 6])
                        fitters.append(kmf)
                        dataframes.append(sub)
                        self.at_risk_data.append([lbl] + [str(self._get_at_risk(kmf, d)) for d in interval_days])
            else:
                # Discrete explicit column matching sets
                for i, group in enumerate(self.km_selected_subgroups):
                    sub = df[df[col].astype(str) == str(group)].copy()
                    if not sub.empty:
                        kmf = KaplanMeierFitter().fit(sub['time'], sub['event'], label=f"{col}={group}")
                        self._add_km_trace(fig, kmf, colors[i % 6], fill_colors[i % 6])
                        fitters.append(kmf)
                        dataframes.append(sub)
                        self.at_risk_data.append([f"{col}={group}"] + [str(self._get_at_risk(kmf, d)) for d in interval_days])

        # Evaluate performance logs and output visual elements
        if fitters:
            prob_strings = []
            med_strings = []
            for kmf in fitters:
                p_val = kmf.predict(days_input)
                prob_strings.append(f"{kmf.label}: {int(p_val * 100)}%")
                m_val = kmf.median_survival_time_ / 30.44
                med_strings.append(f"{kmf.label}: {m_val:.1f} Mo" if not np.isinf(m_val) else f"{kmf.label}: NR")
                
                # Plot intersection tracking dots
                fig.add_trace(go.Scatter(
                    x=[self.end_month], y=[p_val], mode='markers', showlegend=False,
                    marker=dict(color='#ef4444', size=9, line=dict(width=1.5, color='white')),
                    hoverinfo="text", hovertext=f"{kmf.label}<br>Time: {self.end_month} Mo<br>Survival: {int(p_val * 100)}%"
                ))
            
            self.survival_probability_text = " | ".join(prob_strings[:3]) # Bound safe outputs
            self.median_survival_text = " | ".join(med_strings[:3])
            
            if len(fitters) == 1:
                ci_range = fitters[0].confidence_interval_survival_function_
                idx = ci_range.index[np.abs(ci_range.index - days_input).argmin()]
                self.ci_text = f"95% CI: ({ci_range.loc[idx].iloc[0]*100:.1f}% - {ci_range.loc[idx].iloc[1]*100:.1f}%)"
                self.p_value_text = ""
            elif len(fitters) == 2:
                self.ci_text = "Stratified Cohort Group Comparison"
                try:
                    lr_res = logrank_test(dataframes[0]['time'], dataframes[1]['time'], dataframes[0]['event'], dataframes[1]['event'])
                    self.p_value_text = f"Log-rank p: {lr_res.p_value:.4f}"
                except:
                    self.p_value_text = "Log-rank p: N/A"
            else:
                self.ci_text = "Multi-group Stratification Active"
                self.p_value_text = "Log-rank: >2 groups tracked"

        fig.update_layout(
            title={"text": f"<b>Dynamic {self.selected_model} (Custom Dataset)</b>", "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 14}},
            template="simple_white", height=420, autosize=True,
            margin=dict(t=70, b=50, l=40, r=20), legend=dict(orientation="h", y=-0.2),
            modebar={"orientation": "h", "bgcolor": "rgba(0,0,0,0)", "color": "#475569", "activecolor": "#1e293b"}
        )
        fig.update_yaxes(range=[0, 1.05])
        fig.update_xaxes(range=[self.start_month, max_x])
        self.survival_fig = fig

    def _run_cph_engine_isolated(self, df, days_input, ds_name):
        cph = CoxPHFitter().fit(df[['time', 'event'] + self.cph_active_covariates], 'time', 'event')
        summary = cph.summary
        hr_col = [c for c in summary.columns if 'exp(coef)' in c.lower() or 'hazard' in c.lower()][0]
        l_col_name = [c for c in summary.columns if 'lower' in c.lower()][0]
        h_col_name = [c for c in summary.columns if 'upper' in c.lower()][0]
        p_col_name = [c for c in summary.columns if 'p' == c.lower() or 'p-value' in c.lower()][:1][0]

        hr_values = summary[hr_col]
        lower_ci = np.exp(summary[l_col_name]) if 'exp' not in l_col_name.lower() else summary[l_col_name]
        upper_ci = np.exp(summary[h_col_name]) if 'exp' not in h_col_name.lower() else summary[h_col_name]

        self.cph_table_data = [
            [idx, f"{hr:.2f}", f"{low:.2f}", f"{high:.2f}", f"{p:.4f}"] 
            for idx, hr, low, high, p in zip(summary.index, hr_values, lower_ci, upper_ci, summary[p_col_name])
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hr_values, y=summary.index, mode='markers',
            error_x=dict(type='data', symmetric=False, array=upper_ci - hr_values, arrayminus=hr_values - lower_ci, color="#1e293b", thickness=1.5, width=6),
            marker=dict(color='#10b981', size=10, line=dict(width=1, color='white'))
        ))
        fig.add_vline(x=1, line_dash="dash", line_color="#ef4444", opacity=0.8, line_width=2)
        
        dataset_name = "Lung (NCCTG)" if "Lung" in self.selected_dataset else ("Breast Cancer (GBSG2)" if "Breast" in self.selected_dataset else "Custom Cohort")
        
        fig.update_layout(
            title={"text": f"<b>{self.selected_model} Prognosis ({dataset_name})</b>", "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 14}},
            template="simple_white",
            xaxis=dict(type="log", range=[np.log10(0.1), np.log10(10)], tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10]), 
            yaxis=dict(autorange="reversed"), 
            margin=dict(l=100, r=20, t=70, b=40), 
            height=420,
            autosize=True,
            modebar={"orientation": "h", "bgcolor": "rgba(0,0,0,0)", "color": "#475569", "activecolor": "#1e293b"}
        )
        self.cph_fig = fig

        prob_series = cph.predict_survival_function(df.mean())
        closest_time_index = (prob_series.index.to_series() - days_input).abs().idxmin()
        surv_prob_val = prob_series.loc[closest_time_index].values[0]
        
        self.cph_survival_prob = f"{int(surv_prob_val * 100)}%"
        self.cph_log_rank_p = f"LR Test p: {cph.log_likelihood_ratio_test().p_value:.4f}"

    def _run_km_engine(self, df, mapping, days_input, max_x, ds_name, intervals):
        self.at_risk_data = []
        fig = go.Figure()
        interval_days = [i * 30.44 for i in intervals]
        self.at_risk_columns = ["Group"] + [f"M{i}" for i in intervals]
        colors, fill_colors = ["#ec4899", "#3b82f6"], ["rgba(236, 72, 153, 0.12)", "rgba(59, 130, 246, 0.12)"]
        fitters, dataframes = [], []

        valid_active = [c for c in self.active_controls if c in mapping]
        active_splits = [c for c in valid_active if "Split" in c]

        if active_splits:
            split_name = active_splits[0]
            col, _ = mapping[split_name]
            thresh = {"age": self.age_threshold, "tsize": self.size_threshold, "pnodes": self.node_threshold}.get(col, 50)
            cats = [c for c in valid_active if "Split" not in c]
            base_df = df.copy()
            if cats:
                c_col, c_val = mapping[cats[0]]
                base_df = df[df[c_col].astype(str) == str(c_val)].copy()
            
            for i, (cond, lbl) in enumerate([(base_df[col] <= thresh, f"≤{thresh}"), (base_df[col] > thresh, f">{thresh}")]):
                sub = base_df[cond].copy()
                if not sub.empty:
                    kmf = KaplanMeierFitter().fit(sub['time'], sub['event'], label=f"{col.upper()} {lbl}")
                    self._add_km_trace(fig, kmf, colors[i], fill_colors[i])
                    fitters.append(kmf); dataframes.append(sub)
                    self.at_risk_data.append([kmf.label] + [str(self._get_at_risk(kmf, d)) for d in interval_days])
        elif not valid_active:
            kmf = KaplanMeierFitter().fit(df['time'], df['event'], label="Baseline")
            self._add_km_trace(fig, kmf, "#475569", "rgba(71, 85, 105, 0.1)")
            fitters.append(kmf)
            self.at_risk_data = [["Baseline"] + [str(self._get_at_risk(kmf, d)) for d in interval_days]]
        else:
            for i, ctrl in enumerate(valid_active[:2]):
                c, v = mapping[ctrl]
                sub = df[df[c] == float(v)].copy() if pd.api.types.is_numeric_dtype(df[c]) else df[df[c].astype(str) == str(v)].copy()
                if not sub.empty:
                    kmf = KaplanMeierFitter().fit(sub['time'], sub['event'], label=ctrl)
                    self._add_km_trace(fig, kmf, colors[i], fill_colors[i])
                    fitters.append(kmf); dataframes.append(sub)
                    self.at_risk_data.append([ctrl] + [str(self._get_at_risk(kmf, d)) for d in interval_days])

        if fitters:
            if len(fitters) == 2:
                p1, p2 = fitters[0].predict(days_input), fitters[1].predict(days_input)
                self.survival_probability_text = f"{fitters[0].label}: {int(p1 * 100)}% | {fitters[1].label}: {int(p2 * 100)}%"
                m1, m2 = fitters[0].median_survival_time_ / 30.44, fitters[1].median_survival_time_ / 30.44
                self.median_survival_text = f"{fitters[0].label}: {m1:.1f} Mo | {fitters[1].label}: {m2:.1f} Mo"
                self.ci_text = "Comparison Mode Active"
                
                for i, kmf in enumerate(fitters):
                    prob_at_target = kmf.predict(days_input)
                    fig.add_trace(go.Scatter(
                        x=[self.end_month], y=[prob_at_target], mode='markers', showlegend=False,
                        marker=dict(color='#ef4444', size=10, line=dict(width=2, color='white')),
                        hoverinfo="text", hovertext=f"{kmf.label}<br>Time: {self.end_month} Mo<br>Survival: {int(prob_at_target * 100)}%"
                    ))
            else:
                prob = fitters[0].predict(days_input)
                self.survival_probability_text = f"{fitters[0].label}: {int(prob * 100)}%"
                med_mo = fitters[0].median_survival_time_ / 30.44
                self.median_survival_text = f"{med_mo:.1f} Mo" if not np.isinf(med_mo) else "NR"
                ci_range = fitters[0].confidence_interval_survival_function_
                idx = ci_range.index[np.abs(ci_range.index - days_input).argmin()]
                self.ci_text = f"95% CI: ({ci_range.loc[idx].iloc[0]*100:.1f}% - {ci_range.loc[idx].iloc[1]*100:.1f}%)"
                
                fig.add_trace(go.Scatter(
                    x=[self.end_month], y=[prob], mode='markers', showlegend=False,
                    marker=dict(color='#ef4444', size=10, line=dict(width=2, color='white')),
                    hoverinfo="text", hovertext=f"Time: {self.end_month} Mo<br>Survival: {int(prob * 100)}%"
                ))

        self.p_value_text = f"Log-rank p: {logrank_test(dataframes[0]['time'], dataframes[1]['time'], dataframes[0]['event'], dataframes[1]['event']).p_value:.4f}" if len(dataframes) == 2 else ""
        dataset_name = "Lung (NCCTG)" if "Lung" in self.selected_dataset else ("Breast Cancer (GBSG2)" if "Breast" in self.selected_dataset else "Custom Cohort")
        
        fig.update_layout(
            title={"text": f"<b>{self.selected_model} Prognosis ({dataset_name})</b>", "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 14}},
            template="simple_white", height=420, autosize=True,
            margin=dict(t=70, b=50, l=40, r=20), legend=dict(orientation="h", y=-0.2),
            modebar={"orientation": "h", "bgcolor": "rgba(0,0,0,0)", "color": "#475569", "activecolor": "#1e293b"}
        )
        fig.update_yaxes(range=[0, 1.05])
        fig.update_xaxes(range=[self.start_month, max_x])
        self.survival_fig = fig

    def _add_km_trace(self, fig, kmf, color, fill):
        x = (kmf.survival_function_.index / 30.44).tolist()
        y = kmf.survival_function_.iloc[:, 0].tolist()
        ci = kmf.confidence_interval_
        fig.add_trace(go.Scatter(x=x + x[::-1], y=ci.iloc[:, 1].tolist() + ci.iloc[:, 0].tolist()[::-1], fill='toself', fillcolor=fill, line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=y, name=kmf.label, mode='lines', line=dict(color=color, width=2, shape='hv')))

    def _get_at_risk(self, kmf, d_val):
        mask = kmf.event_table.index >= d_val
        return int(kmf.event_table.at[kmf.event_table.index[mask][0], 'at_risk']) if any(mask) else 0

# --- UI Components ---
def control_item(label):
    return rx.checkbox(label, checked=State.active_controls.contains(label), on_change=lambda c: State.handle_km_selection(label, c), size="1")

def km_sidebar_controls():
    return rx.fragment(
        rx.cond(
            State.selected_dataset == "Own CSV File",
            # --- NEW DYNAMIC KM CONTROLS ---
            rx.accordion.item(
                header="Dynamic Group Stratification", value="km_custom_strat",
                content=rx.vstack(
                    rx.text("Isolate Cohort Subgroup:", font_size="0.75em", font_weight="bold"),
                    rx.select(
                        State.csv_columns, placeholder="Select Column...",
                        value=State.km_stratify_col, on_change=State.handle_km_custom_stratify_change,
                        size="1", width="100%"
                    ),
                    rx.cond(
                        State.km_stratify_col,
                        rx.vstack(
                            rx.cond(
                                State.km_is_numeric_stratify,
                                rx.vstack(
                                    rx.text("Continuous Split Value (Median):", font_size="0.7em"),
                                    rx.input(value=State.km_numeric_split_value, on_change=State.update_km_numeric_split, size="1", width="100%"),
                                    align_items="start", width="100%"
                                ),
                                rx.vstack(
                                    rx.text("Select Categorical Subgroups (Max 2):", font_size="0.7em"),
                                    rx.foreach(
                                        State.km_available_subgroups,
                                        lambda sub: rx.checkbox(
                                            sub, checked=State.km_selected_subgroups.contains(sub),
                                            on_change=lambda c: State.update_km_custom_subgroups(sub, c),
                                            size="1"
                                        )
                                    ),
                                    align_items="start"
                                )
                            ),
                            margin_top="0.5em", align_items="start", width="100%"
                        )
                    ),
                    align_items="start", width="100%", spacing="2"
                )
            ),
            # Default Baseline Standard Accordions
            rx.fragment(
                rx.accordion.item(
                    header="Patient Factors", value="p", 
                    content=rx.vstack(
                        rx.cond(
                            State.selected_dataset == "Lung (NCCTG)", 
                            rx.vstack(
                                rx.hstack(control_item("Female"), control_item("Male"), spacing="4"), 
                                rx.text("ECOG Performance:", font_size="0.7em", margin_top="0.5em"), 
                                rx.hstack(control_item("ECOG 0"), control_item("ECOG 1"), spacing="3"), 
                                rx.hstack(control_item("ECOG 2"), control_item("ECOG 3"), spacing="3"), 
                                align_items="start"
                            ), 
                            rx.cond(
                                State.selected_dataset == "Breast Cancer (GBSG2)",
                                rx.vstack(
                                    rx.text("Hormone Therapy:", font_size="0.7em"),
                                    rx.hstack(control_item("Hormone Therapy: Yes"), control_item("Hormone Therapy: No"), spacing="4"),
                                    rx.divider(margin_y="0.5em"),
                                    rx.hstack(control_item("Post-menopausal"), control_item("Pre-menopausal"), spacing="4")
                                ),
                                rx.fragment()
                            )
                        ),
                        rx.divider(margin_y="0.5em"),
                        control_item("Age Split"), 
                        rx.cond(State.active_controls.contains("Age Split"), rx.input(on_blur=State.set_age_threshold, placeholder=f"Age ({State.age_threshold})", size="1")), 
                        align_items="start"
                    )
                ),
                rx.cond(
                    State.selected_dataset == "Breast Cancer (GBSG2)", 
                    rx.accordion.item(
                        header="Biological Factors", value="b", 
                        content=rx.vstack(
                            control_item("Grade 1"), control_item("Grade 2"), control_item("Grade 3"), 
                            rx.divider(), 
                            control_item("Tumor Size Split"), 
                            rx.cond(State.active_controls.contains("Tumor Size Split"), rx.input(on_blur=State.set_size_threshold, placeholder="Size", size="1")), 
                            control_item("Lymph Node Split"), 
                            rx.cond(State.active_controls.contains("Lymph Node Split"), rx.input(on_blur=State.set_node_threshold, placeholder="Nodes", size="1")), 
                            align_items="start"
                        )
                    )
                )
            )
        )
    )

def settings_sidebar():
    return rx.card(
        rx.vstack(
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
                    padding="10px 14px", border_radius="var(--radius-2)", width="100%", margin_bottom="0.5em"
                )
            ),          

            rx.text("MODEL SETTINGS", font_weight="bold", font_size="0.9em"),
            rx.accordion.root(
                rx.accordion.item(
                    header="Data Selection", value="ds", 
                    content=rx.radio_group(["Lung (NCCTG)", "Breast Cancer (GBSG2)", "Own CSV File"], value=State.selected_dataset, on_change=State.set_selected_dataset, size="1")
                ),
                rx.accordion.item(
                    header="Model Selection", value="ms", 
                    content=rx.radio_group(["Kaplan-Meier Survival", "Cox Proportional Hazards"], value=State.selected_model, on_change=State.set_selected_model, size="1")
                ),
                rx.cond(
                    State.selected_model == "Kaplan-Meier Survival", 
                    km_sidebar_controls(),
                    rx.accordion.item(
                        header="CPH Multivariate Covariates", value="cph_cov", 
                        content=rx.vstack(
                            rx.cond(
                                State.selected_dataset == "Own CSV File",
                                rx.fragment(rx.foreach(State.csv_columns, lambda cov: rx.checkbox(cov, on_change=lambda c: State.update_cph_covariates(cov, c), checked=State.cph_active_covariates.contains(cov), size="1"))),
                                rx.cond(
                                    State.selected_dataset == "Breast Cancer (GBSG2)",
                                    rx.fragment(rx.foreach(["age", "menostat", "tsize", "pnodes", "progrec", "estrec", "horTh", "tgrade"], lambda cov: rx.checkbox(cov, on_change=lambda c: State.update_cph_covariates(cov, c), checked=State.cph_active_covariates.contains(cov), size="1"))),
                                    rx.fragment(rx.foreach(["age", "sex", "ph.ecog", "ph.karno", "wt.loss"], lambda cov: rx.checkbox(cov, on_change=lambda c: State.update_cph_covariates(cov, c), checked=State.cph_active_covariates.contains(cov), size="1")))
                                )
                            ),
                            spacing="2", align_items="start"
                        )
                    )
                ),
                type="multiple", variant="ghost", width="100%",
            ),
            
            rx.cond(
                State.selected_dataset == "Own CSV File",
                rx.vstack(
                    rx.upload(
                        rx.hstack(
                            rx.icon("upload", size=14, color=rx.color("blue", 9)), 
                            rx.text("Upload Data File", font_size="0.8em", color=rx.color("blue", 9)), 
                            spacing="2", justify_content="center", align_items="center", height="100%", width="100%"
                        ),
                        id="upload_csv", border=f"1px solid {rx.color('blue', 8)}", background_color=rx.color("blue", 2), border_radius="var(--radius-2)", padding="0", height="28px", cursor="pointer", width="100%",
                        style={"_hover": {"background_color": rx.color("blue", 3)}}
                    ),
                    rx.button(
                        "Process File", on_click=State.handle_upload(rx.upload_files(upload_id="upload_csv")), border=f"1px solid {rx.color('blue', 8)}", background_color=rx.color("blue", 2), color=rx.color("blue", 9), border_radius="var(--radius-2)", height="28px", cursor="pointer", width="100%",
                        style={"_hover": {"background_color": rx.color("blue", 3)}}
                    ),
                    spacing="2", width="100%"
                )
            ),

            rx.divider(),
            rx.text(f"Cohort Timeline: {State.start_month} to {State.end_month} Mo", font_size="0.8em", font_weight="bold"),
            rx.slider(value=State.months_list, on_change=State.set_months_list, min=0, max=State.max_months_limit, width="100%"),
            
            rx.button("RUN ANALYSIS", on_click=State.run_analysis, color_scheme="blue", cursor="pointer", width="100%"),
            spacing="3", align_items="start",
        ),
        width="100%"
    )

def index():                    
    plotly_config = {
        "responsive": True,
        "displaylogo": False,
        "displayModeBar": "always",
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", 
            "autoScale2d", "resetScale2d", "hoverClosestCartesian", 
            "hoverCompareCartesian", "toggleSpikelines"
        ]
    }
    
    return rx.center(
        rx.vstack(
            rx.heading("Clinical Prognosis Platform (CPP)", size="7", color="black", text_align="center", width="100%", margin_bottom="1em"),
            
            # --- SCHEMA PREVIEW CARD ---
            # --- REPLACED DATA DEFINITIONS CARD ---
            rx.cond(
                State.show_schema_preview,
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.hstack(
                                rx.icon("table-properties", size=16, color=rx.color("blue", 9)),
                                rx.text(
                                    f"Dataset Inspector: {State.uploaded_filename}", 
                                    font_weight="bold", 
                                    font_size="0.9em", 
                                    color=rx.color("blue", 11)
                                ),
                                spacing="2"
                            ),
                            rx.button(
                                "Hide Inspector", 
                                size="1", 
                                variant="ghost", 
                                on_click=State.toggle_schema_view, 
                                cursor="pointer"
                            ),
                            width="100%", justify="between", align_items="center"
                        ),
                        
                        rx.tabs.root(
                            rx.tabs.list(
                                rx.tabs.trigger("Data Rows Preview", value="preview"),
                                rx.tabs.trigger("Column Definitions (Schema)", value="schema"),
                            ),
                            rx.tabs.content(
                                # --- TAB 1: SCROLLABLE DATA ROWS PREVIEW WINDOW ---
                                rx.box(
                                    rx.table.root(
                                        rx.table.header(
                                            rx.table.row(
                                                rx.foreach(
                                                    State.data_preview_cols, 
                                                    lambda col: rx.table.column_header_cell(
                                                        col, font_size="11px", font_weight="bold", white_space="nowrap"
                                                    )
                                                )
                                            )
                                        ),
                                        rx.table.body(
                                            rx.foreach(
                                                State.data_preview_rows, 
                                                lambda row: rx.table.row(
                                                    rx.foreach(row, lambda cell: rx.table.cell(cell, font_size="11px", white_space="nowrap"))
                                                )
                                            )
                                        ),
                                        variant="surface", size="1", width="100%"
                                    ),
                                    max_height="240px",   # Constrains height so it doesn't push down the app layout
                                    overflow_y="scroll",  # Empowers user to scroll up/down through rows safely
                                    overflow_x="auto",    # Handles horizontal scroll if columns expand out
                                    border=f"1px solid {rx.color('slate', 4)}",
                                    border_radius="var(--radius-1)",
                                    width="100%",
                                    margin_top="0.5em"
                                ),
                                value="preview",
                            ),
                            rx.tabs.content(
                                # --- TAB 2: ORIGINAL DATA DEFINITIONS (SCHEMA) ---
                                rx.box(
                                    rx.table.root(
                                        rx.table.header(
                                            rx.table.row(
                                                rx.foreach(State.schema_cols, lambda col: rx.table.column_header_cell(col, font_size="11px", font_weight="bold"))
                                            )
                                        ),
                                        rx.table.body(
                                            rx.foreach(State.schema_rows, lambda row: rx.table.row(
                                                rx.foreach(row, lambda cell: rx.table.cell(cell, font_size="11px"))
                                            ))
                                        ),
                                        variant="surface", size="1", width="100%"
                                    ),
                                    margin_top="0.5em"
                                ),
                                value="schema",
                            ),
                            value=State.inspector_active_tab,
                            on_change=State.change_inspector_tab,
                            width="100%"
                        ),
                        spacing="2", width="100%"
                    ),
                    width="100%", max_width="1005px", border=f"1px solid {rx.color('blue', 4)}", margin_bottom="1em"
                )
            ),

            rx.flex(
                # LEFT COLUMN CONTAINER (Plots, Tables, Metrics)
                rx.vstack(
                    rx.card(
                        rx.cond(
                            State.results_ready,
                            rx.vstack(
                                rx.box(
                                    rx.tooltip(
                                        rx.button(
                                            rx.icon(tag="download", size=15), 
                                            on_click=State.notify_future_release,
                                            position="absolute", top="4px", right="52px", z_index="50", size="1", variant="ghost", color_scheme="gray",
                                            style={
                                                "opacity": "0.3", "color": "#475569", "transition": "all 0.2s ease-in-out",
                                                "_hover": {"opacity": "1.0", "color": "#1e293b", "background": "transparent"}
                                            }
                                        ),
                                        content="Download Report", delay_duration=200,
                                    ),
                                    rx.box(
                                        rx.plotly(data=State.survival_fig, config=plotly_config, width="100%"),
                                        display=rx.cond(State.selected_model == "Kaplan-Meier Survival", "block", "none"),
                                        width="100%",
                                    ),
                                    rx.box(
                                        rx.plotly(data=State.cph_fig, config=plotly_config, width="100%"),
                                        display=rx.cond(State.selected_model == "Cox Proportional Hazards", "block", "none"),
                                        width="100%",
                                    ),
                                    width="100%", min_height="420px", max_width="620px", position="relative",
                                ),
                                    
                                rx.box(
                                    rx.cond(
                                        State.selected_model == "Cox Proportional Hazards", 
                                        rx.table.root(
                                            rx.table.header(rx.table.row(rx.foreach(State.cph_table_cols, lambda col: rx.table.column_header_cell(col, font_size="10px")))), 
                                            rx.table.body(rx.foreach(State.cph_table_data, lambda row: rx.table.row(rx.foreach(row, lambda cell: rx.table.cell(cell, font_size="10px"))))), 
                                            variant="surface", size="1", width="100%"
                                        ),
                                        rx.table.root(
                                            rx.table.header(rx.table.row(rx.foreach(State.at_risk_columns, lambda col: rx.table.column_header_cell(col, font_size="10px")))), 
                                            rx.table.body(rx.foreach(State.at_risk_data, lambda row: rx.table.row(rx.foreach(row, lambda cell: rx.table.cell(cell, font_size="10px"))))), 
                                            variant="surface", size="1", width="100%"
                                        )
                                    ),
                                    width="100%", min_height="65px"
                                ),
                                spacing="4", width="100%"
                            ),
                            
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
                                    rx.heading("Awaiting Model Configuration", size="4", color=rx.color("slate", 12), margin_bottom="0.4em"),
                                    rx.text(
                                        "Select your patient cohort and analytical model settings, then click ",
                                        rx.text.strong("RUN ANALYSIS", color=rx.color("blue", 9)),
                                        " to evaluate comparative time-to-event trajectories.",
                                        color=rx.color("slate", 11), font_size="0.85em", text_align="center", style={"textWrap": "balance"} 
                                    ),
                                    spacing="2", align_items="center",
                                    style={
                                        "background": "linear-gradient(135deg, rgba(248, 250, 252, 0.92) 0%, rgba(241, 245, 249, 0.92) 100%)",
                                        "backdrop-filter": "blur(8px)", "padding": "40px", "border-radius": "16px",
                                        "border": "1px solid rgba(226, 232, 240, 0.9)", "max-width": "500px",
                                        "box-shadow": "0 10px 25px -5px rgba(148, 163, 184, 0.05)"
                                    }
                                ),
                                width="620px", height="495px", border=f"1px dashed {rx.color('slate', 5)}",
                                border_radius="var(--radius-3)", class_name="clinical-blueprint-canvas", padding="2em", position="relative"
                            )
                        ),
                        width="100%", padding="16px"
                    ),  
                    
                    rx.card(
                        rx.hstack(
                            rx.vstack(
                                rx.text("SURVIVAL PROB.", font_size="0.7em", color_scheme="blue"), 
                                rx.vstack(rx.foreach(State.survival_prob_lines, lambda line: rx.text(line, font_size="0.8em", font_weight="bold")), spacing="1", align_items="center"),
                                align_items="center", flex="1"
                            ), 
                            rx.divider(orientation="vertical"), 
                            rx.vstack(
                                rx.text("MEDIAN SURVIVAL", font_size="0.7em", color_scheme="blue"), 
                                rx.vstack(rx.foreach(State.median_survival_lines, lambda line: rx.text(line, font_size="0.8em", font_weight="bold")), spacing="1", align_items="center"),
                                align_items="center", flex="1"
                            ), 
                            rx.divider(orientation="vertical"), 
                            rx.vstack(
                                rx.text("STATISTICAL PARAMETERS", font_size="0.7em", color_scheme="blue"), 
                                rx.text(State.ci_text, font_size="0.8em", font_weight="bold"), 
                                rx.text(rx.cond(State.selected_model == "Kaplan-Meier Survival", State.p_value_text, State.cph_log_rank_p), font_size="0.8em", font_weight="bold"), 
                                align_items="center", flex="1.5"
                            ), 
                            width="100%", height="110px"
                        ), 
                        width="100%", bg=rx.color("blue", 2)
                    ),
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("info", size=16, color=rx.color("amber", 9)),
                                rx.text("Clinical Decision Support (CDS) Guidance", font_weight="bold", font_size="0.85em", color=rx.color("amber", 11)),
                                spacing="2", align_items="center"
                            ),
                            rx.text(
                                "Based on the generated cohort demographics, consider assessing alternative therapeutic pathways if median survival falls below critical thresholds.",
                                font_size="0.8em", color=rx.color("slate", 11)
                            ),
                            spacing="2", align_items="start"
                        ),
                        width="100%", border=f"1px solid {rx.color('amber', 5)}", background_color=rx.color("amber", 2), border_radius="var(--radius-3)"
                    ),
                    width="100%", max_width="660px", spacing="4", flex="1 1 0%", 
                ),  
                
                # RIGHT COLUMN CONTAINER (Sidebar panel)
                rx.box(settings_sidebar(), width="320px", flex_shrink="0"),

                # Locked side-by-side display settings
                width="100%", gap="24px", direction="row", justify="center", align_items="start",
            ),

            rx.text(State.metadata_info, font_size="0.8em", color="black"),    
            rx.text(State.cohort_note, font_size="0.8em", color="red"),

            width="100%", max_width="1200px", padding="2em"
        ),  
        width="100%",
    )
 
app = rx.App(
    stylesheets=["style.css"],
    style={
        ".rt-AccordionTrigger": {
            "border_radius": "var(--radius-2) !important",
        },
        ".AccordionTrigger": {
            "border_radius": "var(--radius-2) !important",
        },
        "button[class*='AccordionTrigger']": {
            "border_radius": "var(--radius-2) !important",
        }
    }
)
app.add_page(index, title="Clinical Prognosis Platform (CPP)", route="/")