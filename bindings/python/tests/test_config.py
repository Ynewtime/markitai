"""Public configuration behavior exercised against the real native extension."""

import json
import os
import unittest
from unittest.mock import patch

from markitai import ConversionError, MarkitaiConfig, _native
from markitai.config import (
    CloudflareConfig, DomainProfileConfig, EnvVarNotFoundError, FetchPolicyConfig,
    JinaConfig, LLMConfig, LiteLLMParams, ModelConfig, OutputConfig, PresetConfig,
)


class ConfigurationTests(unittest.TestCase):
    def test_native_normalizes_nested_defaults_coercions_and_extra_fields(self):
        raw = {"llm": {"concurrency": "7", "model_list": [
            {"model_name": "default", "litellm_params": {"model": "test", "weight": "2"}}
        ]}, "image": {"compress": "off"}, "unknown": "ignored"}
        envelope = json.loads(_native.config_json(json.dumps(raw)))
        self.assertIs(envelope["ok"], True)
        value = envelope["result"]
        self.assertEqual(value["llm"]["concurrency"], 7)
        self.assertEqual(value["llm"]["model_list"][0]["litellm_params"]["weight"], 2)
        self.assertIsNone(value["llm"]["model_list"][0]["model_info"])
        self.assertIs(value["image"]["compress"], False)
        self.assertNotIn("unknown", value)
        self.assertEqual(MarkitaiConfig(**raw).model_dump(), value)

    def test_nested_models_lists_and_maps_are_mutable_with_typed_access(self):
        cfg = MarkitaiConfig(llm={"model_list": [{"model_name": "default", "litellm_params": {"model": "first"}}]},
                             presets={"custom": {"llm": True}}, fetch={"domain_profiles": {"example.com": {}}})
        self.assertIsInstance(cfg.llm, LLMConfig)
        self.assertIsInstance(cfg.llm.model_list, list)
        self.assertIsInstance(cfg.llm.model_list[0], ModelConfig)
        self.assertIsInstance(cfg.llm.model_list[0].litellm_params, LiteLLMParams)
        self.assertIsInstance(cfg.presets, dict)
        self.assertIsInstance(cfg.presets["custom"], PresetConfig)
        self.assertIsInstance(cfg.fetch.domain_profiles["example.com"], DomainProfileConfig)
        cfg.llm.model_list[0].litellm_params.model = "second"
        cfg.llm.model_list.append(ModelConfig(model_name="backup", litellm_params=LiteLLMParams(model="third")))
        cfg.presets["custom"].ocr = True
        self.assertEqual([item.litellm_params.model for item in cfg.llm.model_list], ["second", "third"])
        self.assertTrue(cfg.model_dump()["presets"]["custom"]["ocr"])

    def test_assignment_preserves_original_model_rules(self):
        cfg = MarkitaiConfig()
        with self.assertRaises(ValueError):
            cfg.llm.concurreny = 3
        with self.assertRaises(ValueError):
            cfg.typo = True
        cfg.llm.concurrency = "not validated during assignment"
        self.assertEqual(cfg.llm.concurrency, "not validated during assignment")
        self.assertEqual(cfg.llm.model_fields_set, {"concurrency"})
        with self.assertRaises(ValueError):
            MarkitaiConfig.model_validate(cfg.model_dump())
        self.assertIs(MarkitaiConfig.model_validate(cfg), cfg)
        llm = LLMConfig(enabled=True)
        wrapped = MarkitaiConfig(llm=llm)
        self.assertIs(wrapped.llm, llm)
        self.assertEqual(wrapped.model_dump(exclude_unset=True), {"llm": {"enabled": True}})

    def test_model_copy_shares_shallow_sections_and_isolates_deep_sections(self):
        cfg = MarkitaiConfig()
        shallow = cfg.model_copy()
        deep = cfg.model_copy(deep=True)
        self.assertIs(shallow.llm, cfg.llm)
        self.assertIsNot(deep.llm, cfg.llm)
        deep.llm.enabled = True
        self.assertFalse(cfg.llm.enabled)
        replacement = OutputConfig(dir="custom")
        updated = cfg.model_copy(update={"output": replacement})
        self.assertIs(updated.output, replacement)
        self.assertIsNone(cfg.output.dir)
        self.assertIsInstance(dict(cfg)["llm"], LLMConfig)

    def test_serialization_filters_and_roundtrip(self):
        cfg = MarkitaiConfig(output={"dir": "文档"}, llm={"enabled": True, "model_list": [
            {"model_name": "default", "litellm_params": {"model": "test", "api_key": "private"}}
        ]})
        self.assertEqual(cfg.model_dump(include={"output": {"dir"}}), {"output": {"dir": "文档"}})
        filtered = cfg.model_dump(exclude={"llm": {"model_list": {"__all__": {"litellm_params": {"api_key"}}}}})
        self.assertNotIn("api_key", filtered["llm"]["model_list"][0]["litellm_params"])
        self.assertEqual(cfg.output.model_dump(exclude_defaults=True), {"dir": "文档"})
        self.assertEqual(MarkitaiConfig().model_dump(exclude_defaults=True), {})
        self.assertEqual(cfg.model_dump(exclude_unset=True)["output"], {"dir": "文档"})
        self.assertNotIn("profile", cfg.output.model_dump(exclude_none=True))
        self.assertEqual(MarkitaiConfig.model_validate_json(cfg.model_dump_json(indent=2)), cfg)
        self.assertIn("文档", cfg.model_dump_json())

    def test_schema_is_independent_and_has_nested_types_bounds_and_required_fields(self):
        schema = MarkitaiConfig.model_json_schema()
        self.assertEqual(schema["$defs"]["ImageConfig"]["properties"]["quality"]["maximum"], 100)
        self.assertIn("model_name", schema["$defs"]["ModelConfig"]["required"])
        schema["properties"].clear()
        self.assertIn("output", MarkitaiConfig.model_json_schema()["properties"])
        custom = LLMConfig.model_json_schema(ref_template="/schemas/{model}")
        self.assertEqual(custom["properties"]["model_list"]["items"]["$ref"], "/schemas/ModelConfig")

    def test_standalone_nested_models_use_native_custom_validation(self):
        self.assertEqual(LLMConfig(concurrency="3").concurrency, 3)
        self.assertEqual(LiteLLMParams(model="test").weight, 1)
        with self.assertRaises(ValueError):
            LiteLLMParams()
        with self.assertRaises(ValueError):
            FetchPolicyConfig(strategy_priority=[])
        with self.assertRaises(ValueError):
            FetchPolicyConfig(local_only_patterns=["192.0.2.0/99"])
        with self.assertRaises(ValueError):
            DomainProfileConfig(strategy_priority=["static", "static"])
        with self.assertRaises(ValueError):
            LLMConfig.model_validate({"concurrency": "3"}, strict=True)
        with self.assertRaises(ValueError):
            LLMConfig.model_validate({"unknown": 1}, extra="forbid")
        self.assertFalse(LLMConfig.model_validate({"unknown": 1}).enabled)

    def test_native_failures_raise_by_code_and_a_panic_is_a_conversion_error(self):
        failure = json.loads(_native.config_json("[]"))["error"]
        self.assertEqual(failure, {"code": "config_error", "message": "config must be an object"})
        with self.assertRaisesRegex(ValueError, "is not a valid integer|concurrency"):
            LLMConfig(concurrency="many")
        # A native panic answers with the shared envelope, as conversion does.
        panic = json.dumps({"ok": False, "error": {
            "code": "internal_error", "message": "Native configuration failed unexpectedly"}})
        with patch.object(_native, "config_json", return_value=panic):
            with self.assertRaises(ConversionError) as caught:
                LLMConfig(concurrency="3")
        self.assertEqual(caught.exception.code, "internal_error")
        self.assertEqual(str(caught.exception), "Native configuration failed unexpectedly")

    def test_explicit_environment_references_do_not_fall_back_on_missing(self):
        with patch.dict(os.environ, {"JINA_API_KEY": "fallback", "EMPTY": "", "CLOUDFLARE_ACCOUNT_ID": "account"}, clear=True):
            self.assertEqual(JinaConfig().get_resolved_api_key(), "fallback")
            self.assertIsNone(JinaConfig(api_key="env:MISSING").get_resolved_api_key())
            self.assertEqual(JinaConfig(api_key="env:EMPTY").get_resolved_api_key(), "")
            self.assertEqual(CloudflareConfig().get_resolved_account_id(), "account")
            self.assertIsNone(CloudflareConfig(account_id="env:MISSING").get_resolved_account_id())
            with self.assertRaises(EnvVarNotFoundError):
                LiteLLMParams(model="test", api_key="env:MISSING").get_resolved_api_key()


if __name__ == "__main__":
    unittest.main()
