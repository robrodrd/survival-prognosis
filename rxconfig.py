import reflex as rx

config = rx.Config(
    app_name="survival_prognosis",
    plugins=[
        rx.plugins.SitemapPlugin(),
        # Move your theme here
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="light", 
                has_background=True, 
                accent_color="blue"
            )
        ),
    ],
)