import reflex as rx

config = rx.Config(
    app_name="survival_prognosis",
    deploy_url="https://survival-prognosis-service-100651041936.us-central1.run.app",
    plugins=[
        rx.plugins.SitemapPlugin(), # Remove the exclude_paths argument here
    ],
)