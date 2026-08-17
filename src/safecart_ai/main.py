from fastapi import FastAPI

from safecart_ai import __version__
from safecart_ai.api.routes import router

app = FastAPI(
    title="SafeCart AI",
    version=__version__,
    description="Internal product identity matching and inference service.",
)
app.include_router(router)
