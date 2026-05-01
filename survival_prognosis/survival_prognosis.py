import reflex as rx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_lung, load_gbsg2
from lifelines.statistics import logrank_test

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

class State(rx.State):
    # Core Application State
    selected_dataset: str = "Breast Cancer (GBSG2)"
    selected_model: str = "Kaplan-Meier Survival"
    months_list: list[float] = [0.0, 36.0]

    # --- KM Isolated State ---
    active_controls: list[str] = []
    age_threshold: int = 50
    size_threshold: int = 20
    node_threshold: int = 3
    survival_fig: go.Figure = go.Figure()
    at_risk_columns: list[str] = []
    at_risk_data: list[list[str]] = []
    p_value_text: str = ""
    median_survival_text: str = "---"
    survival_probability_text: str = "Awaiting..."
    ci_text: str = "---"

    # --- CPH Isolated State ---
    cph_active_covariates: list[str] = ["age", "sex", "ph.ecog"]
    cph_fig: go.Figure = go.Figure()
    cph_survival_prob: str = "---"
    cph_log_rank_p: str = ""
    cph_table_data: list[list[str]] = []
    cph_table_cols: list[str] = ["Variable", "HR", "95% CI Lower", "95% CI Upper", "p-value"]
    
    cohort_note: str = "Select a dataset and run analysis."

    # --- Setters & Handlers ---
    def set_selected_model(self, val: str): self.selected_model = val
    def set_months_list(self, val: list[float]): self.months_list = val
    
    def set_age_threshold(self, v): 
        try: self.age_threshold = int(v) if v else 50
        except: self.age_threshold = 50
        
    def set_size_threshold(self, v): 
        try: self.size_threshold = int(v) if v else 20
        except: self.size_threshold = 20
        
    def set_node_threshold(self, v): 
        try: self.node_threshold = int(v) if v else 3
        except: self.node_threshold = 3

    def update_cph_covariates(self, covariate: str, checked: bool):
        if checked:
            if covariate not in self.cph_active_covariates: self.cph_active_covariates.append(covariate)
        else:
            if covariate in self.cph_active_covariates: self.cph_active_covariates.remove(covariate)

    def handle_km_selection(self, item: str, checked: bool):
        if checked:
            is_ecog_item = "ECOG" in item
            existing_ecog = [i for i in self.active_controls if "ECOG" in i]
            existing_others = [i for i in self.active_controls if "ECOG" not in i]

            if is_ecog_item:
                if existing_others:
                    self.active_controls = [item]
                else:
                    if len(self.active_controls) >= 2:
                        self.active_controls.pop(1) 
                    self.active_controls.append(item)
            else:
                if not existing_ecog:
                    if len(self.active_controls) >= 2:
                        self.active_controls.pop(1)
                    self.active_controls.append(item)
        else:
            if item in self.active_controls:
                self.active_controls.remove(item)

    def set_selected_dataset(self, val: str):
        self.selected_dataset = val
        self.active_controls = []
        self.p_value_text = ""
        self.ci_text = "---"
        max_limit = 45 if "Lung" in val else 90
        if self.months_list[1] > max_limit:
            self.months_list = [0.0, float(max_limit)]

    @rx.var
    def start_month(self) -> int: return int(self.months_list[0])

    @rx.var
    def end_month(self) -> int: return int(self.months_list[1])

    @rx.var
    def slider_max(self) -> int: return 45 if "Lung" in self.selected_dataset else 90

    @rx.var
    def survival_prob_lines(self) -> list[str]:
        return self.survival_probability_text.split(" | ")

    @rx.var
    def median_survival_lines(self) -> list[str]:
        return self.median_survival_text.split(" | ")

    def calculate_prognosis(self):
        try:
            # We calculate survival probability for the END of the range
            days_input = float(self.end_month * 30.44)

            if self.selected_model == "Kaplan-Meier Survival":
                if "Breast" in self.selected_dataset:
                    df = load_gbsg2(); df['event'] = df['cens'].astype(int)
                    self._run_km_engine(df, BREAST_MAP, days_input, 90, "GBSG2 Breast", [0, 12, 24, 36, 48, 60, 72, 84])
                    self.cohort_note = "Study: German Breast Cancer Study Group (GBSG2, 1994). Parameters: 686 patients with primary node-positive breast cancer."
                else:
                    df = load_lung().dropna(subset=['time', 'status', 'sex', 'ph.ecog'])
                    df['event'] = (df['status'] - 1).astype(int)
                    self._run_km_engine(df, LUNG_MAP, days_input, 45, "NCCTG Lung", [0, 10, 20, 30, 40])
                    self.cohort_note = "Study: NCCTG Lung Cancer Data (Loprinzi et al., 1994). Parameters: Advanced lung cancer survival vs. ECOG performance."
            else:
                if self.selected_dataset == "Lung (NCCTG)":
                    df_cph = load_lung().dropna(subset=['time', 'status'] + self.cph_active_covariates)
                    df_cph['event'] = (df_cph['status'] - 1).astype(int)
                    self._run_cph_engine_isolated(df_cph, days_input, "NCCTG Lung")
                else:
                    df_cph = load_gbsg2().dropna()
                    df_cph['event'] = df_cph['cens'].astype(int)
                    self._run_cph_engine_isolated(df_cph, days_input, "GBSG2 Breast")
        except Exception as e:
            print(f"Main Engine Error: {e}")

    def _run_cph_engine_isolated(self, df, days_input, ds_name):
        try:
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
            fig.update_layout(template="simple_white", xaxis=dict(type="log", range=[np.log10(0.1), np.log10(10)], tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10]), yaxis=dict(autorange="reversed"), margin=dict(l=120, r=40, t=50, b=40), height=450)
            self.cph_fig = fig

            prob_series = cph.predict_survival_function(df.mean())
            closest_time_index = (prob_series.index.to_series() - days_input).abs().idxmin()
            surv_prob_val = prob_series.loc[closest_time_index].values[0]
            
            self.cph_survival_prob = f"{int(surv_prob_val * 100)}%"
            self.cph_log_rank_p = f"LR Test p: {cph.log_likelihood_ratio_test().p_value:.4f}"
        except Exception as e:
            print(f"CPH Engine Error: {e}")

    def _run_km_engine(self, df, mapping, days_input, max_x, ds_name, intervals):
        try:
            fig = go.Figure()
            interval_days = [i * 30.44 for i in intervals]
            self.at_risk_columns = ["Group"] + [f"M{i}" for i in intervals]
            colors, fill_colors = ["#ec4899", "#3b82f6"], ["rgba(236, 72, 153, 0.12)", "rgba(59, 130, 246, 0.12)"]
            fitters, dataframes, self.at_risk_data = [], [], []

            active_splits = [c for c in self.active_controls if "Split" in c]
            if active_splits:
                split_name = active_splits[0]
                col, _ = mapping[split_name]
                thresh = {"age": self.age_threshold, "tsize": self.size_threshold, "pnodes": self.node_threshold}.get(col, 50)
                cats = [c for c in self.active_controls if "Split" not in c]
                base_df = df.copy()
                if cats:
                    c_col, c_val = mapping[cats[0]]
                    base_df = df[df[c_col].astype(str) == str(c_val)].copy()
                
                for i, (cond, lbl) in enumerate([(base_df[col] <= thresh, f"≤{thresh}"), (base_df[col] > thresh, f">{thresh}")]):
                    sub = base_df[cond].copy()
                    if not sub.empty:
                        kmf = KaplanMeierFitter().fit(sub['time'], sub['event'], label=f"{col.upper()} {lbl}")
                        self._add_km_trace(fig, kmf, colors[i], fill_colors[i])
                        y_val = float(kmf.predict(days_input))
                        # FIX: Use end_month for the marker X position
                        fig.add_trace(go.Scatter(
                            x=[float(self.end_month)], y=[y_val], mode='markers',
                            name=f"Point {lbl}", marker=dict(color='#ef4444', size=10, line=dict(color='white', width=2)),
                            showlegend=False, hoverinfo="skip"
                        ))
                        fitters.append(kmf); dataframes.append(sub)
                        self.at_risk_data.append([kmf.label] + [str(self._get_at_risk(kmf, d)) for d in interval_days])

            elif not self.active_controls:
                kmf = KaplanMeierFitter().fit(df['time'], df['event'], label="Baseline")
                self._add_km_trace(fig, kmf, "#475569", "rgba(71, 85, 105, 0.1)")
                y_val = kmf.predict(days_input)
                # FIX: Use end_month
                fig.add_trace(go.Scatter(x=[self.end_month], y=[y_val], mode='markers', marker=dict(color='#ef4444', size=10, line=dict(color='white', width=2)), showlegend=False))
                fitters.append(kmf)
                self.at_risk_data = [["Baseline"] + [str(self._get_at_risk(kmf, d)) for d in interval_days]]

            else:
                for i, ctrl in enumerate(self.active_controls[:2]):
                    c, v = mapping[ctrl]
                    sub = df[df[c] == float(v)].copy() if pd.api.types.is_numeric_dtype(df[c]) else df[df[c].astype(str) == str(v)].copy()
                    if not sub.empty:
                        kmf = KaplanMeierFitter().fit(sub['time'], sub['event'], label=ctrl)
                        self._add_km_trace(fig, kmf, colors[i], fill_colors[i])
                        y_val = kmf.predict(days_input)
                        # FIX: Use end_month
                        fig.add_trace(go.Scatter(x=[self.end_month], y=[y_val], mode='markers', marker=dict(color='#ef4444', size=10, line=dict(color='white', width=2)), showlegend=False))
                        fitters.append(kmf); dataframes.append(sub)
                        self.at_risk_data.append([ctrl] + [str(self._get_at_risk(kmf, d)) for d in interval_days])

            if fitters:
                if len(fitters) == 2:
                    p1, p2 = fitters[0].predict(days_input), fitters[1].predict(days_input)
                    self.survival_probability_text = f"{fitters[0].label}: {int(p1 * 100)}% | {fitters[1].label}: {int(p2 * 100)}%"
                    m1, m2 = fitters[0].median_survival_time_ / 30.44, fitters[1].median_survival_time_ / 30.44
                    self.median_survival_text = f"{fitters[0].label}: {m1:.1f} Mo | {fitters[1].label}: {m2:.1f} Mo"
                    self.ci_text = "Comparison Mode: 95% CI Active"
                else:
                    prob = fitters[0].predict(days_input)
                    self.survival_probability_text = f"{fitters[0].label}: {int(prob * 100)}%"
                    med_mo = fitters[0].median_survival_time_ / 30.44
                    self.median_survival_text = f"{med_mo:.1f} Mo" if not np.isinf(med_mo) else "NR"
                    ci_range = fitters[0].confidence_interval_survival_function_
                    idx = ci_range.index[np.abs(ci_range.index - days_input).argmin()]
                    self.ci_text = f"95% CI: ({ci_range.loc[idx].iloc[0]*100:.1f}% - {ci_range.loc[idx].iloc[1]*100:.1f}%)"

            if len(dataframes) == 2:
                res = logrank_test(dataframes[0]['time'], dataframes[1]['time'], dataframes[0]['event'], dataframes[1]['event'])
                self.p_value_text = f"Log-rank p: {res.p_value:.4f}"
            else:
                self.p_value_text = ""

            fig.update_layout(template="simple_white", height=430, margin=dict(t=50, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.25))
            fig.update_yaxes(range=[0, 1.05])
            fig.update_xaxes(range=[self.start_month, max_x])
            self.survival_fig = fig

        except Exception as e:
            print(f"KM Engine Error: {e}")

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
            header="Patient Factors", 
            value="p", 
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
                    rx.vstack(
                        rx.text("Hormone Therapy:", font_size="0.7em"),
                        rx.hstack(control_item("Hormone Therapy: Yes"), control_item("Hormone Therapy: No"), spacing="4"),
                        rx.divider(margin_y="0.5em"),
                        rx.hstack(control_item("Post-menopausal"), control_item("Pre-menopausal"), spacing="4")
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
                header="Biological Factors", 
                value="b", 
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
                rx.accordion.item(header="Data Selection", value="ds", content=rx.radio_group(["Lung (NCCTG)", "Breast Cancer (GBSG2)", "Own CSV File"], value=State.selected_dataset, on_change=State.set_selected_dataset, size="1")),
                rx.accordion.item(header="Model Selection", value="ms", content=rx.radio_group(["Kaplan-Meier Survival", "Cox Proportional Hazards"], value=State.selected_model, on_change=State.set_selected_model, size="1")),
                rx.cond(State.selected_model == "Kaplan-Meier Survival", km_sidebar_controls(), rx.accordion.item(header="CPH Multivariate Covariates", value="cph_cov", content=rx.vstack(rx.foreach(["age", "sex", "ph.ecog", "ph.karno", "wt.loss"], lambda cov: rx.checkbox(cov, on_change=lambda c: State.update_cph_covariates(cov, c), checked=State.cph_active_covariates.contains(cov), size="1")), spacing="2", align_items="start"))),
                type="multiple", variant="ghost", width="100%",
            ),
            rx.divider(),
            rx.text(f"Cohort Timeline: {State.start_month} to {State.end_month} Mo", font_size="0.8em", font_weight="bold"),
            rx.slider(value=State.months_list, on_change=State.set_months_list, min=0, max=State.slider_max, width="100%"),
            rx.button("RUN ANALYSIS", on_click=State.calculate_prognosis, color_scheme="blue", width="100%"),
            spacing="3", align_items="start",
        ),
        width="100%"
    )

def index():
    return rx.center(
        rx.vstack(
            rx.heading("Clinical Analytics Platform", size="8", margin_bottom="0.5em", width="100%", text_align="center"),
            rx.flex(
                # Left Column
                rx.vstack(
                    rx.card(
                        rx.vstack(
                            rx.text(f"{State.selected_model} Prognosis ({State.selected_dataset})", font_weight="bold", font_size="1.1em", color=rx.color("slate", 11), width="100%", text_align="center", margin_top="1.5em"),
                            rx.cond(State.selected_model == "Kaplan-Meier Survival", rx.plotly(data=State.survival_fig, width="100%"), rx.plotly(data=State.cph_fig, width="100%")),
                            rx.cond(State.selected_model == "Kaplan-Meier Survival", 
                                rx.table.root(
                                    rx.table.header(rx.table.row(rx.foreach(State.at_risk_columns, lambda col: rx.table.column_header_cell(col, font_size="10px")))), 
                                    rx.table.body(rx.foreach(State.at_risk_data, lambda row: rx.table.row(rx.foreach(row, lambda cell: rx.table.cell(cell, font_size="10px"))))), 
                                    variant="surface", size="1", width="100%"
                                ), 
                                rx.table.root(
                                    rx.table.header(rx.table.row(rx.foreach(State.cph_table_cols, lambda col: rx.table.column_header_cell(col, font_size="10px")))), 
                                    rx.table.body(rx.foreach(State.cph_table_data, lambda row: rx.table.row(rx.foreach(row, lambda cell: rx.table.cell(cell, font_size="10px"))))), 
                                    variant="surface", size="1", width="100%"
                                )
                            ),
                            spacing="4", align_items="center"
                        ), 
                        width="100%"
                    ),
                    rx.card(
                        rx.hstack(
                            rx.vstack(
                                rx.text("SURVIVAL PROB.", font_size="0.7em", color_scheme="blue"), 
                                rx.cond(State.selected_model == "Kaplan-Meier Survival",
                                    rx.vstack(
                                        rx.foreach(State.survival_prob_lines, lambda line: rx.text(line, font_size="0.8em", font_weight="bold")),
                                        spacing="1", align_items="center"
                                    ),
                                    rx.heading(State.cph_survival_prob, size="5")
                                ),
                                align_items="center", flex="1"
                            ), 
                            rx.divider(orientation="vertical"), 
                            rx.vstack(
                                rx.text("MEDIAN SURVIVAL", font_size="0.7em", color_scheme="blue"), 
                                rx.cond(State.selected_model == "Kaplan-Meier Survival",
                                    rx.vstack(
                                        rx.foreach(State.median_survival_lines, lambda line: rx.text(line, font_size="0.8em", font_weight="bold")),
                                        spacing="1", align_items="center"
                                    ),
                                    rx.heading("N/A (CPH)", size="5")
                                ),
                                align_items="center", flex="1"
                            ), 
                            rx.divider(orientation="vertical"), 
                            rx.vstack(
                                rx.text("STATISTICAL PARAMETERS", font_size="0.7em", color_scheme="blue"), 
                                rx.text(rx.cond(State.selected_model == "Kaplan-Meier Survival", State.ci_text, "95% Confidence Intervals:"), font_size="0.8em", font_weight="bold"), 
                                rx.text(rx.cond(State.selected_model == "Kaplan-Meier Survival", State.p_value_text, State.cph_log_rank_p), font_size="0.8em", font_weight="bold"), 
                                align_items="center", flex="1.5"
                            ), 
                            width="100%", height="110px"
                        ), 
                        width="100%", bg=rx.color("blue", 2)
                    ),
                    flex="2", spacing="4"
                ),
                rx.box(settings_sidebar(), flex="1"),
                width="100%", spacing="6"
            ),
            rx.text(State.cohort_note, font_size="0.7em", color="gray", padding="1em"),
            width="100%", max_width="1250px", padding="2em"
        )
    )

app = rx.App(theme=rx.theme(accent_color="blue", radius="large"))
app.add_page(index)