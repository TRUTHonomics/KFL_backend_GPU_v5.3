"""ConfigurableResource voor KFL backfill runs."""

from dagster import ConfigurableResource
from pydantic import Field


class BackfillRunConfig(ConfigurableResource):
    """Configuratie voor KFL backfill runs.
    
    Deze resource maakt alle backfill parameters configureerbaar in de Dagster Launchpad.
    """
    
    asset_id: int = Field(
        default=1,
        description="Asset ID voor backfill. Gebruikt door: kfl/indicators_and_signals, alle MTF assets",
    )
    
    mode: str = Field(
        default="gaps_only",
        description="Backfill mode: 'full' of 'gaps_only'. Gebruikt door: kfl/indicators_and_signals",
    )
    
    intervals: str = Field(
        default="1,60,240,D",
        description="Komma-gescheiden intervallen (bijv. '1,60,240,D'). Gebruikt door: kfl/indicators_and_signals, alle MTF assets",
    )
    
    start_date: str = Field(
        default="",
        description="Optionele startdatum (YYYY-MM-DD). Gebruikt door: kfl/indicators_and_signals",
    )
    
    end_date: str = Field(
        default="",
        description="Optionele einddatum (YYYY-MM-DD). Gebruikt door: kfl/indicators_and_signals",
    )
    
    use_gpu: bool = Field(
        default=True,
        description="GPU acceleration aan/uit. Gebruikt door: kfl/indicators_and_signals",
    )
