"""Unit tests for the encoder helper. No real model weights."""
from models.embed.encoders import ENCODER_REGISTRY, encode


class FakeModel:
    """Echoes the texts it received so tests can assert prefix application."""
    def __init__(self):
        self.received: list[str] = []

    def encode(self, texts, **kw):
        self.received = list(texts)
        return self.received   # the loop under test does not consume the return value


def test_e5_prefix_applied():
    m = FakeModel()
    encode(m, ["story A", "story B"],
           task="Retrieve stories with a similar narrative to the given story.")
    assert m.received == [
        "Instruct: Retrieve stories with a similar narrative to the given story.\nQuery: story A",
        "Instruct: Retrieve stories with a similar narrative to the given story.\nQuery: story B",
    ]


def test_no_prefix_when_task_is_none():
    m = FakeModel()
    encode(m, ["story A"], task=None)
    assert m.received == ["story A"]


def test_normalize_embeddings_kw_passed():
    seen_kwargs: dict = {}

    class CapturingModel:
        def encode(self, texts, **kw):
            seen_kwargs.update(kw)
            return texts

    encode(CapturingModel(), ["x"], task=None)
    assert seen_kwargs["normalize_embeddings"] is True
    assert seen_kwargs["convert_to_numpy"] is True


def test_registry_contract():
    for key, value in ENCODER_REGISTRY.items():
        assert isinstance(key, str) and key
        assert isinstance(value, tuple) and len(value) == 2
        model_id, task = value
        assert isinstance(model_id, str) and model_id
        assert task is None or isinstance(task, str)
    assert "e5_mistral" in ENCODER_REGISTRY
    assert "sbert_mpnet" in ENCODER_REGISTRY


def test_story_emb_registered():
    assert "story_emb" in ENCODER_REGISTRY
    model_id, task = ENCODER_REGISTRY["story_emb"]
    assert model_id == "uhhlt/story-emb"
    # StoryEmbed paper §3 confirms the same E5 instruction prefix is used.
    assert task == ENCODER_REGISTRY["e5_mistral"][1]


def test_story_emb_conditions_override():
    from models.embed.encoders import ENCODER_CONDITIONS
    assert ENCODER_CONDITIONS["story_emb"] == ("raw_text",)
    # e5/sbert have no override — default (all six) applies via .get() at call site.
    assert "e5_mistral" not in ENCODER_CONDITIONS
    assert "sbert_mpnet" not in ENCODER_CONDITIONS
