from __future__ import annotations

from flux.health import log_event
from flux.retrieval import flux_store_ex
from flux.visualization import chronicle_data
from mocks import MockEmbeddingBackend


def test_chronicle_heat_from_recent_retrievals(store):
    emb = MockEmbeddingBackend()
    grain1, _ = flux_store_ex(
        "Mercury surface temperatures swing between deep cold and intense heat.",
        store=store,
        llm=None,
        emb=emb,
    )
    grain2, _ = flux_store_ex(
        "Consistent hashing keeps distributed cache keys stable during resharding.",
        store=store,
        llm=None,
        emb=emb,
    )

    log_event(store, "retrieval", "grains_returned", {"grain_ids": [grain1]})

    data = chronicle_data(store)
    by_id = {grain[0]: grain for grain in data["grains"]}

    assert data["grain_fields"][-1] == "heat"
    assert by_id[grain1][-1] > 0
    assert by_id[grain2][-1] == 0
