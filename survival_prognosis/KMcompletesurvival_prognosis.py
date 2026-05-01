import reflex as rx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_lung, load_gbsg2
from lifelines.statistics import logrank_test

# Mapping for categorical variables
LUNG_MAP = {"Female": ("sex", 1), "Male": ("sex", 2), "Age Split": ("age", "split")}
BREAST_MAP = {
    "Hormone Therapy: Yes": ("horTh", "yes"),
    "Hormone Therapy: No": ("horTh", "no"),
    "Post-menopausal": ("menostat", "Post"),
    "Pre-menopausal": ("menostat", "Pre"),
    "Grade 1": ("tgrade", "I"),
    "Grade 2": ("tgrade", "II"),
    "Grade 3": ("tgrade", "III"),
    "Age Split": ("age", "split"),
    "Tumor Size Split": ("tsize", "split"),
    "Lymph Node Split": ("pnodes", "split"),
    "Progesterone Split": ("progrec", "split"),
    "Estrogen Split": ("estrec", "split"),
}

class State(rx.State):
    selected_dataset: str = "Breast Cancer (GBSG2)"
    selected_model: str = "Kaplan-Meier Survival"
    months_list: list[float] = [36.0]
    active_controls: list[str] = []
   
    # Thresholds
    age_threshold: int = 50
    size_threshold: int = 20
    node_threshold: int = 3  
    prog_threshold: int = 20
    est_threshold: int = 20

    # UI Result States
    survival_probability_text: str = "Awaiting..."
    median_survival_text: str = "---"
    ci_text: str = "---"
    p_value_text: str = ""  
    survival_fig: go.Figure = go.Figure()
    at_risk_columns: list[str] = []
    at_risk_data: list[list[str]] = []
    cohort_note: str = "Select a dataset and run analysis to view cohort details."

    # --- Threshold Setters ---
    def set_age_threshold(self, v): self.age_threshold = int(v) if v else 50
    def set_size_threshold(self, v): self.size_threshold = int(v) if v else 20
    def set_node_threshold(self, v): self.node_threshold = int(v) if v else 3
    def set_prog_threshold(self, v): self.prog_threshold = int(v) if v else 20
    def set_est_threshold(self, v): self.est_threshold = int(v) if v else 20

    def set_selected_model(self, val: str): self.selected_model = val
    def set_months_list(self, val: list[float]): self.months_list = val

    def set_selected_dataset(self, val: str):
        self.selected_dataset = val
        self.active_controls = []
        self.p_value_text = ""
        self.ci_text = "---"
        max_limit = 45 if "Lung" in val else 90
        if self.months_list[0] > max_limit:
            self.months_list = [float(max_limit)]
        if val == "Custom (Upload CSV)":
            self.cohort_note = "Awaiting custom data source. No PHI will be stored on servers."

    def handle_selection(self, item: str, checked: bool):
        if checked:
            if len(self.active_controls) < 3: self.active_controls.append(item)
        else:
            if item in self.active_controls: self.active_controls.remove(item)

    @rx.var
    def current_months(self) -> int:
        return int(self.months_list[0])

    @rx.var
    def slider_max(self) -> int:
        return 45 if "Lung" in self.selected_dataset else 90

    def calculate_prognosis(self):
        if self.selected_dataset == "Custom (Upload CSV)":
            return
       
        try:
            days_input = float(self.current_months * 30.44)
           
            # 1. LOAD DATA
            if self.selected_dataset == "Breast Cancer (GBSG2)":
                df = load_gbsg2()
                df['event'] = df['cens'].astype(int)
                mapping = BREAST_MAP
                intervals = [0, 12, 24, 36, 48, 60, 72, 84]
                max_x = 90
                ds_name = "GBSG2 Breast Cancer"
                self.cohort_note = "Cohort: German Breast Cancer Study Group (GBSG2). Prospective trial. (n=686)"
            else:
                df = load_lung().dropna(subset=['time', 'status', 'sex'])
                df['event'] = (df['status'] - 1).astype(int)
                mapping = LUNG_MAP
                intervals = [0, 10, 20, 30, 40]
                max_x = 45
                ds_name = "NCCTG Lung Cancer"
                self.cohort_note = "Cohort: North Central Cancer Treatment Group (NCCTG). (n=228)"

            # 2. MODEL ENGINE SELECTION
            if "Kaplan" in self.selected_model:
                self._run_km_engine(df, mapping, days_input, max_x, ds_name, intervals)
            else:
                self._run_cph_engine(df, days_input, ds_name)

        except Exception as e:
            self.survival_probability_text = "Error"
            print(f"Analysis Error: {e}")

    def _run_cph_engine(self, df, days_input, ds_name):
        """Cox Proportional Hazards Engine: Multivariate Risk Analysis."""
        cph_df = df.copy()
       
        # Binary encoding for CPH compatibility
        if "Lung" in ds_name:
            cph_df['sex'] = cph_df['sex'].map({1: 0, 2: 1}) # Male=0, Female=1
            cols = ['time', 'event', 'age', 'sex', 'ph.ecog']
        else:
            cph_df['horTh'] = cph_df['horTh'].map({'no': 0, 'yes': 1})
            cph_df['menostat'] = cph_df['menostat'].map({'Pre': 0, 'Post': 1})
            cph_df['tgrade'] = cph_df['tgrade'].map({'I': 1, 'II': 2, 'III': 3})
            cols = ['time', 'event', 'age', 'tsize', 'pnodes', 'horTh', 'menostat', 'tgrade']
       
        cph_df = cph_df[cols].dropna()
        cph = CoxPHFitter()
        cph.fit(cph_df, duration_col='time', event_col='event')
       
        # FOREST PLOT VISUALIZATION
        summary = cph.summary
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=summary['exp(coef)'], y=summary.index, mode='markers',
            error_x=dict(type='data', array=summary['exp(coef) upper 0.95'] - summary['exp(coef)'],
                         arrayminus=summary['exp(coef)'] - summary['exp(coef) lower 0.95']),
            marker=dict(color='#3b82f6', size=10), name="Hazard Ratio"
        ))
        fig.add_vline(x=1, line_dash="dash", line_color="red")
        fig.update_layout(title=f"CPH Hazard Ratios: {ds_name}", template="simple_white", xaxis_title="Hazard Ratio (95% CI)")
       
        # Statistics
        prob = cph.predict_survival_function(cph_df.mean()).loc[days_input][0]
        self.survival_fig = fig
        self.survival_probability_text = f"{int(prob * 100)}%"
        self.median_survival_text = "N/A (CPH)"
        self.ci_text = "Multivariate Adjusted"
        self.p_value_text = f"LR Test p: {cph.log_likelihood_ratio_test().p_value:.4f}"
        self.at_risk_data = [] # Reset table for CPH

    def _run_km_engine(self, df, mapping, days_input, max_x, ds_name, intervals):
        """Original Kaplan-Meier Engine: Group Comparison (Ground Truth)."""
        fig = go.Figure()
        interval_days = [i * 30.44 for i in intervals]
        self.at_risk_columns = ["Group"] + [f"M{i}" for i in intervals]
        colors, fill_colors = ["#ec4899", "#3b82f6"], ["rgba(236, 72, 153, 0.12)", "rgba(59, 130, 246, 0.12)"]
        fitters, dataframes, self.at_risk_data = [], [], []

        active_splits = [c for c in self.active_controls if "Split" in c]
       
        if active_splits:
            split_name = active_splits[0]
            col, _ = mapping[split_name]
            thresh_map = {"age": self.age_threshold, "tsize": self.size_threshold, "pnodes": self.node_threshold}
            thresh = thresh_map.get(col, 50)
            cats = [c for c in self.active_controls if "Split" not in c]
            base_df = df.copy()
            prefix = f"{cats[0]}: " if cats else ""
            if cats:
                c_col, c_val = mapping[cats[0]]
                base_df = df[df[c_col].astype(str) == str(c_val)].copy()

            for i, (cond, lbl) in enumerate([(base_df[col] <= thresh, f"≤{thresh}"), (base_df[col] > thresh, f">{thresh}")]):
                sub = base_df[cond].copy()
                full_lbl = f"{prefix}{col.upper()} {lbl}"
                if not sub.empty:
                    kmf = KaplanMeierFitter().fit(sub['time'], sub['event'], label=full_lbl)
                    self.add_manual_trace(fig, kmf, full_lbl, colors[i], fill_colors[i])
                    fitters.append(kmf); dataframes.append(sub)
                    self.at_risk_data.append([full_lbl] + [str(self._get_at_risk(kmf, d)) for d in interval_days])
       
        elif not self.active_controls:
            kmf = KaplanMeierFitter().fit(df['time'], df['event'], label="Baseline")
            self.add_manual_trace(fig, kmf, "Baseline", "#475569", "rgba(71, 85, 105, 0.1)")
            fitters.append(kmf)
            self.at_risk_data = [["Baseline"] + [str(self._get_at_risk(kmf, d)) for d in interval_days]]
        else:
            for i, ctrl in enumerate(self.active_controls[:2]):
                c, v = mapping[ctrl]; sub = df[df[c].astype(str) == str(v)].copy()
                if not sub.empty:
                    kmf = KaplanMeierFitter().fit(sub['time'], sub['event'], label=ctrl)
                    self.add_manual_trace(fig, kmf, ctrl, colors[i], fill_colors[i])
                    fitters.append(kmf); dataframes.append(sub)
                    self.at_risk_data.append([ctrl] + [str(self._get_at_risk(kmf, d)) for d in interval_days])

        if fitters:
            active_kmf = fitters[0]
            prob = active_kmf.predict(days_input)
            self.survival_probability_text = f"{int(prob * 100)}%"
            med_mo = active_kmf.median_survival_time_ / 30.44
            self.median_survival_text = f"{med_mo:.1f} Mo" if not np.isinf(med_mo) else "NR"
            ci_range = active_kmf.confidence_interval_survival_function_
            idx = ci_range.index[np.abs(ci_range.index - days_input).argmin()]
            low, high = ci_range.loc[idx]
            self.ci_text = f"95% CI: ({low*100:.1f}% - {high*100:.1f}%)"
            fig.add_trace(go.Scatter(x=[0, max_x], y=[0.5, 0.5], mode="lines", line=dict(color="#94a3b8", width=1.5, dash="dash"), showlegend=False))
            fig.add_trace(go.Scatter(x=[self.current_months], y=[prob], mode='markers', marker=dict(size=8, color='red', line=dict(width=1, color='white')), showlegend=False))

        if len(dataframes) == 2:
            results = logrank_test(dataframes[0]['time'], dataframes[1]['time'], event_observed_A=dataframes[0]['event'], event_observed_B=dataframes[1]['event'])
            self.p_value_text = f"Log-rank p: {results.p_value:.4f}"

        fig.update_layout(title=dict(text=f"KM Analysis: {ds_name}", x=0.5), template="simple_white", height=430, margin=dict(t=50, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.25))
        fig.update_yaxes(range=[0, 1.05], title="Survival Probability")
        fig.update_xaxes(range=[0, max_x], title="Months")
        self.survival_fig = fig

    def add_manual_trace(self, fig, kmf, label, color, fill_color):
        x = (kmf.survival_function_.index / 30.44).tolist()
        y = kmf.survival_function_.iloc[:, 0].tolist()
        ci = kmf.confidence_interval_
        y_upper, y_lower = ci.iloc[:, 1].tolist(), ci.iloc[:, 0].tolist()
        fig.add_trace(go.Scatter(x=x + x[::-1], y=y_upper + y_lower[::-1], fill='toself', fillcolor=fill_color, line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=y, name=label, line=dict(color=color, width=1.5, shape='hv')))

    def _get_at_risk(self, kmf, d_val):
        mask = kmf.event_table.index >= d_val
        return int(kmf.event_table.at[kmf.event_table.index[mask][0], 'at_risk']) if any(mask) else 0

# --- UI Helpers ---

def control_item(label):
    return rx.checkbox(label, checked=State.active_controls.contains(label), on_change=lambda c: State.handle_selection(label, c), size="1")

def threshold_input(label, setter_func, split_key):
    return rx.cond(
        State.active_controls.contains(split_key),
        rx.vstack(
            rx.text(label, font_size="0.7em", color_scheme="gray"),
            rx.input(on_change=setter_func, type="number", size="1", placeholder="Value..."),
            width="100%", spacing="1", padding_left="1.5em", margin_bottom="0.5em"
        )
    )

def index():
    return rx.center(
        rx.vstack(
            rx.heading("Clinical Survival Analytics", size="8", margin_bottom="0.5em"),
            rx.flex(
                rx.vstack(
                    rx.card(
                        rx.vstack(
                            rx.plotly(data=State.survival_fig, width="100%"),
                            rx.cond(
                                State.selected_model == "Kaplan-Meier Survival",
                                rx.table.root(
                                    rx.table.header(rx.table.row(rx.foreach(State.at_risk_columns, lambda col: rx.table.column_header_cell(col, font_size="10px")))),
                                    rx.table.body(rx.foreach(State.at_risk_data, lambda row: rx.table.row(rx.foreach(row, lambda cell: rx.table.cell(cell, font_size="10px"))))),
                                    variant="surface", size="1", width="100%"
                                )
                            ),
                            spacing="4"
                        ),
                        variant="classic", width="100%"
                    ),
                    rx.card(
                        rx.hstack(
                            rx.vstack(rx.text("SURVIVAL PROB.", font_size="0.7em", color_scheme="blue"), rx.heading(State.survival_probability_text, size="6"), align_items="center", flex="1"),
                            rx.divider(orientation="vertical"),
                            rx.vstack(rx.text("MEDIAN SURVIVAL", font_size="0.7em", color_scheme="blue"), rx.heading(State.median_survival_text, size="6"), align_items="center", flex="1"),
                            rx.divider(orientation="vertical"),
                            rx.vstack(
                                rx.text("STATISTICAL PARAMETERS", font_size="0.7em", color_scheme="blue"),
                                rx.text(State.ci_text, font_size="0.8em", font_weight="bold", color="black"),
                                rx.text(State.p_value_text, font_size="0.8em", font_weight="bold", color="black"),
                                align_items="center", flex="1.5"
                            ),
                            width="100%", height="80px"
                        ),
                        width="100%", bg=rx.color("blue", 2)
                    ),
                    rx.vstack(
                        rx.text("COHORT INFORMATION", font_size="10px", font_weight="bold", color_scheme="gray"),
                        rx.text(State.cohort_note, font_size="11px", color_scheme="gray", italic=True),
                        spacing="1", align_items="start", width="100%", padding_top="0.5em"
                    ),
                    flex="2", spacing="4"
                ),
                rx.vstack(
                    rx.card(
                        rx.vstack(
                            rx.text("MODEL SETTINGS", font_weight="bold", font_size="0.9em", color_scheme="blue", margin_bottom="0.5em"),
                            rx.text("SELECT YOUR DATA", font_weight="bold", font_size="0.7em", color_scheme="gray"),
                            rx.select(["Breast Cancer (GBSG2)", "Lung (NCCTG)", "Custom (Upload CSV)"], value=State.selected_dataset, on_change=State.set_selected_dataset, width="100%"),
                            rx.text("SELECT YOUR MODEL", font_weight="bold", font_size="0.7em", color_scheme="gray", margin_top="0.5em"),
                            rx.select(["Kaplan-Meier Survival", "Cox Proportional Hazards"], value=State.selected_model, on_change=State.set_selected_model, width="100%"),
                            rx.divider(),
                            rx.accordion.root(
                                rx.accordion.item(
                                    header=rx.text("Patient Factors", font_weight="bold", font_size="0.9em"),
                                    value="patient",
                                    content=rx.vstack(
                                        rx.cond(
                                            State.selected_dataset == "Lung (NCCTG)",
                                            rx.vstack(rx.text("Sex", font_size="0.7em", color_scheme="gray"), rx.hstack(control_item("Female"), control_item("Male"), spacing="4"), align_items="start"),
                                            rx.vstack(
                                                rx.text("Hormone Therapy", font_size="0.7em", color_scheme="gray"), rx.hstack(control_item("Hormone Therapy: Yes"), control_item("Hormone Therapy: No"), spacing="4"),
                                                rx.text("Menopause Status", font_size="0.7em", color_scheme="gray"), rx.hstack(control_item("Post-menopausal"), control_item("Pre-menopausal"), spacing="4"),
                                                align_items="start"
                                            )
                                        ),
                                        rx.divider(),
                                        control_item("Age Split"),
                                        threshold_input("Split Age (Years)", State.set_age_threshold, "Age Split"),
                                        align_items="start", spacing="2"
                                    )
                                ),
                                rx.cond(
                                    State.selected_dataset == "Breast Cancer (GBSG2)",
                                    rx.accordion.item(
                                        header=rx.text("Tumor Biology", font_weight="bold", font_size="0.9em"),
                                        value="tumor",
                                        content=rx.vstack(
                                            rx.text("Histological Grade", font_size="0.7em", color_scheme="gray"),
                                            rx.grid(control_item("Grade 1"), control_item("Grade 2"), control_item("Grade 3"), columns="3", spacing="4"),
                                            rx.divider(),
                                            control_item("Tumor Size Split"),
                                            threshold_input("Split Size (mm)", State.set_size_threshold, "Tumor Size Split"),
                                            control_item("Lymph Node Split"),
                                            threshold_input("Split Node Count", State.set_node_threshold, "Lymph Node Split"),
                                            align_items="start", spacing="2"
                                        )
                                    )
                                ),
                                width="100%", type="multiple"
                            ),
                            rx.divider(),
                            rx.text(f"Month Selection: {State.current_months}", font_size="0.8em", font_weight="bold"),
                            rx.slider(value=State.months_list, on_change=State.set_months_list, max=State.slider_max, width="100%"),
                            rx.button("RUN ANALYSIS", on_click=State.calculate_prognosis, color_scheme="blue", width="100%", margin_top="1em"),
                            spacing="3", align_items="start"
                        )
                    ),
                    flex="1", spacing="4"
                ),
                width="100%", spacing="6"
            ),
            width="100%", max_width="1250px", padding="2em"
        )
    )

app = rx.App(theme=rx.theme(accent_color="blue", radius="large"))
app.add_page(index)