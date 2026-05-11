import reflex as rx

config = rx.Config(
    app_name="survival_prognosis",
    # This tells the frontend where the backend API lives
    api_url="https://clinicalprognosis.app", 
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="light", 
                has_background=True, 
                accent_color="blue"
            )
        ),
    ],
)