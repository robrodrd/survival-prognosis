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
    debug_status: str = "Awaiting click..."
    show_dialog: bool = False
    
    # Track whether an analysis run has successfully completed to reveal results panels
    results_ready: bool = False

    def notify_future_release(self):
        return rx.toast("This premium feature will be available in a future release!", position="bottom-right")
    
    def toggle_dialog(self):
        self.show_dialog = not self.show_dialog

    selected_dataset: str = ""  
    selected_model: str = ""

    active_controls: list[str] = rx.field(default_factory=list)
    data_selection: str = "Default"
    metadata_info: str = "Select a dataset to begin." 

    def run_analysis(self):         
        self.debug_status = ""
        
        if not self.selected_model or not self.selected_dataset:
            self.cohort_note = ""
            return rx.toast(
                "Selection Required: Please select a Model and a Dataset first.", 
                position="top-right", 
                close_button=True,
                style={
                    "background-color": "#fff5f5",
                    "color": "#c53030",
                    "border": "1px solid #feb2b2",
                    "font-weight": "bold",
                }
            )

        if self.selected_dataset == "Own CSV File":
            self.show_dialog = False
            self.metadata_info = "Source: User-uploaded CSV. Custom cohort analysis."
            return rx.toast(
                "Custom CSV integration is slated for a future platform release.", 
                position="top-right", 
                close_button=True,
                style={
                    "background-color": "#eff6ff",
                    "color": "#1d4ed8",
                    "border": "1px solid #3b82f6",
                    "font-weight": "bold",
                }
            )

        self.show_dialog = False
        return self.calculate_prognosis()
    
    def reset_state(self):
        self.selected_dataset = ""
        self.selected_model = ""
        self.results_ready = False
        self.metadata_info = "Select a dataset and model to begin analysis."
        self.cohort_note = ""
        
    def update_dataset(self, value: str):
        self.selected_dataset = value
        self.active_controls = [] 
        self.cph_active_covariates = []
        self.results_ready = False  # Instantly hides the layout panels during changes smoothly
        self.survival_probability_text = ""
        self.median_survival_text = "---"
        self.p_value_text = ""
        
        self.at_risk_data = [["-", "-", "-", "-", "-", "-"]] 
        self.cph_table_data = [["-", "-", "-", "-", "-"]]
        self.ci_text = "---"
        self.cohort_note = ""

        if "Breast" in value or "GBSG2" in value:
            self.metadata_info = "Source: GBSG2 Breast Cancer Study. Created: 1994. 686 patients with node-positive cancer."
        elif "Lung" in value or "NCCTG" in value:
            self.metadata_info = "Source: NCCTG Lung Cancer Data (Loprinzi et al.). Created: 1994. Advanced lung cancer study."
        else:
            self.metadata_info = "Source: User-uploaded CSV. Custom cohort analysis."

        return rx.toast(f"{value} Selected.", position="top-right", close_button=True)

    def update_model(self, value: str):
        self.selected_model = value
        self.results_ready = False  # Resets display window to pristine template state safely
        self.survival_probability_text = ""
        self.median_survival_text = "---"
        self.p_value_text = ""
        
        self.at_risk_data = [["-", "-", "-", "-", "-", "-"]]
        self.cph_table_data = [["-", "-", "-", "-", "-"]]
        self.cohort_note = ""
        return rx.toast(f"{value} Selected.", position="top-right", close_button=True)

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

    def update_cph_covariates(self, covariate: str, checked: bool):
        self.results_ready = False
        if checked:
            if covariate not in self.cph_active_covariates: self.cph_active_covariates.append(covariate)
        else:
            if covariate in self.cph_active_covariates: self.cph_active_covariates.remove(covariate)
        self.cohort_note = ""
        return rx.toast("Covariates updated.", position="top-right", close_button=True)

    def handle_upload(self, files: list[rx.UploadFile]):
        for file in files:
            import io
            df = pd.read_csv(io.BytesIO(file.file.read()))
            self.uploaded_data = df
            self.csv_columns = df.columns.tolist()
            self.cph_active_covariates = []
            self.cohort_note = ""
            return rx.toast(f"Uploaded {file.filename}.", position="top-right", close_button=True)

    def handle_km_selection(self, item: str, checked: bool):
        self.results_ready = False 
        self.cohort_note = "Parameters changed."
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

    @rx.var
    def start_month(self) -> int: return int(self.months_list[0])

    @rx.var
    def end_month(self) -> int: return int(self.months_list[1])
        
    @rx.var
    def max_months_limit(self) -> int:
        return 45 if "Lung" in self.selected_dataset else 90
    
    @rx.var 
    def survival_prob_lines(self) -> list[str]:
        return self.survival_probability_text.split(" | ") if self.survival_probability_text else []

    @rx.var
    def median_survival_lines(self) -> list[str]:
        return self.median_survival_text.split(" | ") if self.median_survival_text else []

    def calculate_prognosis(self):      
        if self.selected_dataset == "" or self.selected_model == "":
            return rx.toast("Selection Required: Please choose a Dataset and Model.", position="top-right", close_button=True)

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
                    df = self.uploaded_data.copy()
                    time_col = next((c for c in df.columns if c.lower() == 'time'), 'time')
                    event_col = next((c for c in df.columns if c.lower() in ['event', 'status', 'cens']), 'event')
                    df['time'] = df[time_col]
                    df['event'] = df[event_col]
                    self._run_km_engine(df, {}, days_input, int(df['time'].max()/30.44), "Custom CSV", [0, 12, 24, 36])

            else: 
                if not self.cph_active_covariates:
                    return rx.toast("Configuration Required: Select at least one covariate.", position="top-right", close_button=True)

                if "Lung" in self.selected_dataset:
                    df_cph = load_lung().dropna(subset=['time', 'status'] + self.cph_active_covariates).copy()
                    df_cph['event'] = (df_cph['status'] - 1).astype(int)
                    if 'sex' in self.cph_active_covariates and df_cph['sex'].dtype == object:
                        df_cph['sex'] = df_cph['sex'].map({'Male': 1, 'Female': 2, '1': 1, '2': 2})
                    self._run_cph_engine_isolated(df_cph, days_input, "NCCTG Lung")
                elif "Breast" in self.selected_dataset:
                    df_cph = load_gbsg2().dropna().copy()
                    df_cph['event'] = df_cph['cens'].astype(int)
                    if 'horTh' in self.cph_active_covariates: df_cph['horTh'] = df_cph['horTh'].map({'yes': 1, 'no': 0})
                    if 'menostat' in self.cph_active_covariates: df_cph['menostat'] = df_cph['menostat'].map({'Post': 1, 'Pre': 0})
                    if 'tgrade' in self.cph_active_covariates: df_cph['tgrade'] = df_cph['tgrade'].map({'I': 1, 'II': 2, 'III': 3})
                    self._run_cph_engine_isolated(df_cph, days_input, "GBSG2 Breast")
                else: 
                    df_cph = self.uploaded_data.dropna(subset=['time', 'event'] + self.cph_active_covariates)
                    self._run_cph_engine_isolated(df_cph, days_input, "Custom CSV")

            self.results_ready = True  # Activates presentation state cleanly
            self.cohort_note = ""
            return rx.toast("Analysis Complete!", position="top-right", close_button=True)

        except Exception as e:
            self.results_ready = False
            self.cohort_note = f"Analysis Error: {str(e)}"
            return rx.toast(f"Analysis Error: {str(e)}", position="top-right", close_button=True)

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
                
                # --- RESTORED: Dots for both comparison curves ---
                for i, kmf in enumerate(fitters):
                    prob_at_target = kmf.predict(days_input)
                    fig.add_trace(go.Scatter(
                        x=[self.end_month], y=[prob_at_target],
                        mode='markers', showlegend=False,
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
                
                # --- RESTORED: Dot for single baseline curve ---
                fig.add_trace(go.Scatter(
                    x=[self.end_month], y=[prob],
                    mode='markers', showlegend=False,
                    marker=dict(color='#ef4444', size=10, line=dict(width=2, color='white')),
                    hoverinfo="text", hovertext=f"Time: {self.end_month} Mo<br>Survival: {int(prob * 100)}%"
                ))

        self.p_value_text = f"Log-rank p: {logrank_test(dataframes[0]['time'], dataframes[1]['time'], dataframes[0]['event'], dataframes[1]['event']).p_value:.4f}" if len(dataframes) == 2 else ""
        dataset_name = "Lung (NCCTG)" if "Lung" in self.selected_dataset else ("Breast Cancer (GBSG2)" if "Breast" in self.selected_dataset else "Custom Cohort")
        
        fig.update_layout(
            title={"text": f"<b>{self.selected_model} Prognosis ({dataset_name})</b>", "x": 0.5, "y": 0.92, "xanchor": "center", "font": {"size": 14}},
            template="simple_white", height=420, margin=dict(t=70, b=50, l=40, r=20), legend=dict(orientation="h", y=-0.2)
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
                rx.cond(State.selected_dataset != "Own CSV File", control_item("Age Split")), 
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

def settings_sidebar():
    return rx.card(
        rx.vstack(
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
                                rx.fragment(rx.foreach(["age", "sex", "ph.ecog", "ph.karno", "wt.loss"], lambda cov: rx.checkbox(cov, on_change=lambda c: State.update_cph_covariates(cov, c), checked=State.cph_active_covariates.contains(cov), size="1")))
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
                        rx.hstack(rx.icon("upload", size=16, color=rx.color("blue", 9)), rx.text("Upload target data file", font_size="0.8em"), spacing="2", justify_content="center", width="100%"),
                        id="upload_csv", border=f"1px solid {rx.color('blue', 8)}", background_color=rx.color("blue", 2), padding="0.75em", width="100%"
                    ),
                    rx.button("Process File", on_click=State.handle_upload(rx.upload_files(upload_id="upload_csv")), size="1", variant="surface", width="100%"),
                    spacing="2", width="100%"
                )
            ),
            rx.divider(),
            rx.text(f"Cohort Timeline: {State.start_month} to {State.end_month} Mo", font_size="0.8em", font_weight="bold"),
            rx.slider(value=State.months_list, on_change=State.set_months_list, min=0, max=State.max_months_limit, width="100%"),
            rx.button("RUN ANALYSIS", on_click=State.run_analysis, color_scheme="blue", width="100%"),
            spacing="3", align_items="start",
        ),
        width="100%"
    )

def index():                    
    return rx.center(
        rx.vstack(
            rx.heading("Clinical Prognosis Platform (CPP)", size="7", color="black", text_align="center", width="100%", margin_bottom="1em"),
            
            rx.flex(
                # LEFT COLUMN CONTAINER (Plots, Tables, Metrics)
                rx.vstack(
                    rx.card(
                        # FIX: Control entire result visualization box through State.results_ready switch
                        rx.cond(
                            State.results_ready,
                            rx.vstack(
                                rx.box(
                                    rx.button(
                                        rx.icon(tag="download", size=14), on_click=State.notify_future_release,
                                        position="absolute", top="12px", right="16px", z_index="10", size="1", variant="soft", color_scheme="gray"
                                    ),
                                    rx.box(
                                        rx.plotly(data=State.survival_fig, config={"responsive": True, "displaylogo": False}),
                                        display=rx.cond(State.selected_model == "Kaplan-Meier Survival", "block", "none"),
                                        width="100%",
                                        overflow="hidden"
                                    ),
                                    rx.box(
                                        rx.plotly(data=State.cph_fig, config={"responsive": True, "displaylogo": False}),
                                        display=rx.cond(State.selected_model == "Cox Proportional Hazards", "block", "none"),
                                        width="100%",
                                        overflow="hidden"
                                    ),
                                    width="100%",
                                    min_height="420px",
                                    max_width="620px",
                                    position="relative",
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
                                    width="100%",
                                    min_height="65px"
                                ),
                                spacing="4",
                                width="100%"
                            ),
                            # FIX: Rock-solid, non-collapsible default placeholder when state is uncalculated or cleared
                            rx.center(
                                rx.vstack(
                                    rx.icon("line-chart", size=32, color=rx.color("slate", 7)),
                                    rx.text("Results Area Clean", font_weight="semibold", color=rx.color("slate", 9), font_size="0.95em"),
                                    rx.text("Configure parameters and click 'RUN ANALYSIS' to generate graphs.", color=rx.color("slate", 8), font_size="0.8em", text_align="center"),
                                    spacing="2",
                                    align_items="center"
                                ),
                                width="620px",
                                height="495px", # Perfectly fills out the graph + table grid space to keep spacing uniform
                                border=f"1px dashed {rx.color('slate', 5)}",
                                border_radius="var(--radius-3)",
                                background_color=rx.color("slate", 2),
                                padding="2em"
                            )
                        ),
                        width="100%",
                        padding="16px"
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
                        width="100%",
                        border=f"1px solid {rx.color('amber', 5)}",
                        background_color=rx.color("amber", 2),
                        border_radius="var(--radius-3)"
                    ),
                    width="640px",
                    spacing="4"
                ),
                
                # RIGHT COLUMN CONTAINER (Sidebar panel)
                rx.box(
                    settings_sidebar(), 
                    width="320px"
                ),
                
                width="1000px", 
                spacing="6",
                justify_content="space-between"
            ),

            rx.text(State.metadata_info, font_size="0.8em", color="black"),    
            rx.text(State.cohort_note, font_size="0.8em", color="red"),
            width="1000px", padding="2em"   
        ),
    )

app = rx.App()
app.add_page(index, title="Clinical Prognosis Platform (CPP)", route="/")