import reflex as rx
from reflex_base.plugins.sitemap import SitemapPlugin
from reflex_components_radix.plugin import RadixThemesPlugin
from reflex_components_radix.themes.base import Theme

config = rx.Config(
    app_name="survival_prognosis",
    plugins=[
        SitemapPlugin(),
        RadixThemesPlugin(
            theme=Theme.create(
                appearance="light",
                accent_color="blue",
                gray_color="slate",
                radius="medium",
                scaling="100%"
            )
        )
    ]
)