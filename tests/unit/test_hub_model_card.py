"""Unit tests for alpha_zero.hub.model_card.

Tests cover:
- ModelCardMetadata defaults and tag merging
- Template loading (file-based and fallback)
- Card generation round-trip
- write_model_card file I/O
"""

from alpha_zero.hub.model_card import (
    ModelCardMetadata,
    generate_model_card,
    write_model_card,
)


class TestModelCardMetadata:
    """Tests for ModelCardMetadata dataclass."""

    def test_default_values(self):
        meta = ModelCardMetadata()
        assert meta.model_name == "alpha-zero-civ6-model"
        assert isinstance(meta.license, str)
        assert len(meta.license) > 0
        assert isinstance(meta.library_name, str)
        assert len(meta.library_name) > 0

    def test_custom_values(self):
        meta = ModelCardMetadata(
            model_name="test-model",
            description="A test model.",
            architecture="ResNet",
            num_parameters="1M",
        )
        assert meta.model_name == "test-model"
        assert meta.description == "A test model."
        assert meta.architecture == "ResNet"

    def test_tag_merging_deduplicates(self):
        meta = ModelCardMetadata(tags=("pytorch", "custom-tag"))
        # "pytorch" is in DEFAULT_MODEL_TAGS — should not be duplicated.
        tag_list = list(meta.all_tags)
        assert tag_list.count("pytorch") == 1
        assert "custom-tag" in tag_list

    def test_tag_order_preserved(self):
        meta = ModelCardMetadata(tags=("zzz-tag", "aaa-tag"))
        tags = meta.all_tags
        zzz_idx = tags.index("zzz-tag")
        aaa_idx = tags.index("aaa-tag")
        # Custom tags come after default tags.
        assert zzz_idx < aaa_idx

    def test_tags_yaml_formatting(self):
        meta = ModelCardMetadata(tags=("extra",))
        yaml = meta.tags_yaml()
        assert yaml.startswith("  - ")
        assert "extra" in yaml
        # Each line should be a YAML list item.
        for line in yaml.strip().split("\n"):
            assert line.strip().startswith("- ")

    def test_empty_license_gets_default(self):
        meta = ModelCardMetadata(license="")
        assert meta.license != ""

    def test_explicit_license_preserved(self):
        meta = ModelCardMetadata(license="Apache-2.0")
        assert meta.license == "Apache-2.0"


class TestGenerateModelCard:
    """Tests for generate_model_card function."""

    def test_returns_string(self):
        meta = ModelCardMetadata()
        card = generate_model_card(meta)
        assert isinstance(card, str)
        assert len(card) > 0

    def test_contains_model_name(self):
        meta = ModelCardMetadata(model_name="my-awesome-model")
        card = generate_model_card(meta)
        assert "my-awesome-model" in card

    def test_contains_yaml_frontmatter(self):
        meta = ModelCardMetadata()
        card = generate_model_card(meta)
        assert card.startswith("---")
        assert "library_name:" in card
        assert "tags:" in card
        assert "license:" in card

    def test_contains_usage_example(self):
        meta = ModelCardMetadata(
            import_path="alpha_zero.core",
            class_name="MyModel",
            repo_id="org/repo",
        )
        card = generate_model_card(meta)
        assert "from alpha_zero.core import MyModel" in card
        assert 'from_pretrained("org/repo")' in card

    def test_custom_description_included(self):
        meta = ModelCardMetadata(description="This is a custom description.")
        card = generate_model_card(meta)
        assert "This is a custom description." in card

    def test_metrics_included(self):
        meta = ModelCardMetadata(metrics="| Metric | Value |\n|---|---|\n| ELO | 1500 |")
        card = generate_model_card(meta)
        assert "ELO" in card
        assert "1500" in card


class TestWriteModelCard:
    """Tests for write_model_card function."""

    def test_writes_file(self, tmp_path):
        meta = ModelCardMetadata(model_name="test-write")
        output = tmp_path / "README.md"
        result = write_model_card(meta, output)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "test-write" in content

    def test_creates_parent_dirs(self, tmp_path):
        meta = ModelCardMetadata()
        output = tmp_path / "nested" / "deep" / "README.md"
        result = write_model_card(meta, output)
        assert result.exists()

    def test_returns_resolved_path(self, tmp_path):
        meta = ModelCardMetadata()
        output = tmp_path / "README.md"
        result = write_model_card(meta, output)
        assert result.is_absolute()


class TestTemplateLoading:
    """Tests for template file loading vs fallback."""

    def test_fallback_template_used_when_file_missing(self, tmp_path, monkeypatch):
        """When template file doesn't exist, fallback is used."""
        import alpha_zero.hub.model_card as mc

        monkeypatch.setattr(mc, "CARD_TEMPLATE_PATH", tmp_path / "nonexistent.md")
        meta = ModelCardMetadata()
        card = generate_model_card(meta)
        # Should still produce a valid card from the fallback.
        assert "---" in card
        assert meta.model_name in card

    def test_custom_template_file_used(self, tmp_path, monkeypatch):
        """When template file exists, it should be used."""
        import alpha_zero.hub.model_card as mc

        custom_template = tmp_path / "custom.md.template"
        custom_template.write_text("CUSTOM: $model_name", encoding="utf-8")
        monkeypatch.setattr(mc, "CARD_TEMPLATE_PATH", custom_template)

        meta = ModelCardMetadata(model_name="test-custom-template")
        card = generate_model_card(meta)
        assert card == "CUSTOM: test-custom-template"
