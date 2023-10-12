import asyncio
from collections import ChainMap
import colorsys
import logging
from typing import Any, Callable

from homeassistant.exceptions import TemplateError
from homeassistant.const import EVENT_COMPONENT_LOADED, STATE_UNKNOWN
from homeassistant.core import Event, HomeAssistant, valid_entity_id
from homeassistant.helpers.entity_registry import async_get
from homeassistant.helpers.template import _get_state_if_valid, _RESERVED_NAMES, Template, TemplateEnvironment
from homeassistant.helpers.translation import _TranslationCache, TRANSLATION_FLATTEN_CACHE, TRANSLATION_LOAD_LOCK
from homeassistant.loader import bind_hass

from .const import (DOMAIN, CUSTOM_TEMPLATES_SCHEMA, CONF_PRELOAD_TRANSLATIONS, CONST_EVAL_FUNCTION_NAME,
                    CONST_STATE_TRANSLATED_FUNCTION_NAME, CONST_STATE_ATTR_TRANSLATED_FUNCTION_NAME,
                    CONST_TRANSLATED_FUNCTION_NAME, CONST_ALL_TRANSLATIONS_FUNCTION_NAME, 
                    CONST_HLS_TO_RGB_NAME, CONST_RGB_TO_HLS_NAME,
                    DEFAULT_UNAVAILABLE_STATES, CONST_IS_AVAILABLE_FUNCTION_NAME)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = CUSTOM_TEMPLATES_SCHEMA
ConfigType = dict[str, Any]

class TranslatableTemplate:

    def __init__(self, hass: HomeAssistant, available_languages):
        self._hass = hass
        self._available_languages = available_languages

    def validate_language(self, language):
        if language not in self._available_languages:
            raise TemplateError(f"Language {language} is not loaded")  # type: ignore[arg-type]

class ConvertColorSystem:
    def __init__(self, hass: HomeAssistant):
        self._hass = hass
    def __call__(self, fr: str, to: str, *colors):
        '''
        First ensure the colors received are float and that the correct number of values are received,
            then convert all values to RGB,
            finally change values to the proper system
        '''
        color_mapping = {
            'rgb': {
                'values': 3,
                'from': self._from_rgb,
                'to': self._to_rgb
            },
            'hs': {
                'values': 2,
                'from': self._from_hs,
                'to': self._to_hs
            },
            'hls': {
                'values': 3,
                'from': colorsys.hls_to_rgb,
                'to': colorsys.rgb_to_hls
            },
            'hsv': {
                'values': 3,
                'from': colorsys.hsv_to_rgb,
                'to': colorsys.rgb_to_hsv
            },
            'yiq': {
                'values': 3,
                'from': colorsys.yiq_to_rgb,
                'to': colorsys.rgb_to_yiq
            },
        }
        colors = self._convert_if_not_floats(colors)
        if fr not in color_mapping:
            raise TemplateError(f"Unrecognized color conversion type for from: {fr}.")
        if to not in color_mapping:
            raise TemplateError(f"Unrecognized color conversion type for to: {to}.")
        color_len = color_mapping.get(fr).get('values')
        from_func = color_mapping.get(fr).get('from')
        to_func = color_mapping.get(to).get('to')
        if len(colors) != color_len:
            raise TemplateError(f"Color conversion from {fr} requires {color_len} values, received {len(colors)}.")
        rgb = from_func(*colors)
        result = to_func(*rgb)
        return list(result)

    def _convert_if_not_floats(self, colors):
        '''
        First check if values can be converted to float, otherwise raise error.
        If any value is over 1.0, we assume we received bytes instead of a float between 0 and 1.0 and convert it.
        '''
        if len(colors) == 1 and type(colors[0]) in [set, tuple, list]:
            colors = colors[0]
        try:
            colors = [float(color) for color in colors]
        except ValueError:
            raise TemplateError(f"Expected either int or float for colors, received {type(color)}.")
            
        if sum([color > 1 for color in colors]):
            result = [color/ 255.0 for color in colors]
        else:
            result = colors
        return result

    def _from_rgb(self, *rgb):
        if len(rgb) == 1:
            rgb = list(rgb[0])
        else:
            rgb = list(rgb)
        return rgb

    def _from_hs(self, *hs):
        '''
        Convert to hsv by setting value to 1.0
        '''
        if len(hs) == 1:
            hs = list(hs[0])
        else:
            hs = list(hs)
        hs.append(1.0)
        rgb = colorsys.hsv_to_rgb(*hs)
        return rgb

    def _to_rgb(self, *rgb):
        if len(rgb) == 1:
            rgb = list(rgb[0])
        else:
            rgb = list(rgb)
        return rgb

    def _to_hs(self, *rgb):
        if len(rgb) == 1:
            rgb = list(rgb[0])
        else:
            rgb = list(rgb)
        hsv = colorsys.rgb_to_hsv(*rgb)
        hs = hsv[:-1]
        return hs

class IsAvailable():
    def __init__(self, hass: HomeAssistant):
        self._hass = hass
    def __call__(self, entity_id: str, unavailable_states=DEFAULT_UNAVAILABLE_STATES):
        unavailable_states = [state.lower() if type(state) is str else state for state in unavailable_states]
        if "." in entity_id:
            state = _get_state_if_valid(self._hass, entity_id)

        else:
            if entity_id in _RESERVED_NAMES:
                return None
            if not valid_entity_id(f"{entity_id}.entity"):
                raise TemplateError(f"Invalid domain name '{entity_id}'") 

        if state is not None:
            state = state.state
        if state is str:
            state = state.lower()
        result = state not in unavailable_states
        return result
        
    def __repr__(self):
        return f"<template {self.__class__.__name__}>"

class StateTranslated(TranslatableTemplate):

    def __init__(self, hass: HomeAssistant, available_languages):
        super().__init__(hass, available_languages)

    def __call__(self, entity_id: str, language: str):
        self.validate_language(language)
        state = None
        if "." in entity_id:
            state = _get_state_if_valid(self._hass, entity_id)

        else:
            if entity_id in _RESERVED_NAMES:
                return None

            if not valid_entity_id(f"{entity_id}.entity"):
                raise TemplateError(f"Invalid domain name '{entity_id}'")  # type: ignore[arg-type]

        if state is None:
            return STATE_UNKNOWN
        entry = async_get(self._hass).async_get(entity_id)
        domain = state.domain
        device_class = "_"
        if "device_class" in state.attributes:
            device_class = state.attributes["device_class"]

        translations = get_cached_translations(self._hass, language, "entity_component")
        key = f"component.{domain}.entity_component.{device_class}.state.{state.state}"
        if len(translations) > 0 and key in translations:
            return str(translations[key])
        if (entry is not None and
                entry.unique_id is not None and
                hasattr(entry, "translation_key") and
                entry.translation_key is not None):
            key = f"component.{entry.platform}.entity.{domain}.{entry.translation_key}.state.{state.state}"
            translations = get_cached_translations(self._hass, language, "entity")
        if len(translations) > 0 and key in translations:
            return str(translations[key])

        key = f"component.{domain}.state.{device_class}.{state.state}"
        translations = get_cached_translations(self._hass, language, "state", domain)
        if len(translations) > 0 and key in translations:
            return str(translations[key])
        _LOGGER.warning(f"No translation found for entity: {entity_id}")
        return state.state

    def __repr__(self):
        return "<template StateTranslated>"


class StateAttrTranslated(TranslatableTemplate):

    def __init__(self, hass: HomeAssistant, available_languages):
        super().__init__(hass, available_languages)

    def __call__(self, entity_id: str, attribute: str, language: str):
        self.validate_language(language)
        state = None
        if "." in entity_id:
            state = _get_state_if_valid(self._hass, entity_id)

        else:
            if entity_id in _RESERVED_NAMES:
                return None

            if not valid_entity_id(f"{entity_id}.entity"):
                raise TemplateError(f"Invalid domain name '{entity_id}'")  # type: ignore[arg-type]

        if state is None:
            return STATE_UNKNOWN
        attribute_value = None
        if attribute in state.attributes:
            attribute_value = state.attributes.get(attribute)
        entry = async_get(self._hass).async_get(entity_id)
        domain = state.domain
        device_class = "_"
        if "device_class" in state.attributes:
            device_class = state.attributes["device_class"]

        translations = get_cached_translations(self._hass, language, "entity_component")
        key = f"component.{domain}.entity_component.{device_class}.state_attributes.{attribute}.state.{attribute_value}"
        if len(translations) > 0 and key in translations:
            return str(translations[key])
        if (entry is not None and
                entry.unique_id is not None and
                hasattr(entry, "translation_key") and
                entry.translation_key is not None):
            key = f"component.{entry.platform}.entity.{domain}.{entry.translation_key}.state_attributes.{attribute}.state.{attribute_value}"
            translations = get_cached_translations(self._hass, language, "entity")
        if len(translations) > 0 and key in translations:
            return str(translations[key])
        _LOGGER.warning(f"No translation found for entity: {entity_id} and attribute: {attribute}")
        return attribute_value

    def __repr__(self):
        return "<template StateAttrTranslated>"


class Translated(TranslatableTemplate):

    def __init__(self, hass: HomeAssistant, available_languages):
        super().__init__(hass, available_languages)

    def __call__(self, key: str, language: str):
        self.validate_language(language)
        translations = get_cached_translations(self._hass, language, "state")
        if len(translations) > 0 and key in translations:
            return str(translations[key])
        translations = get_cached_translations(self._hass, language, "entity_component")
        if len(translations) > 0 and key in translations:
            return str(translations[key])
        translations = get_cached_translations(self._hass, language, "entity")
        if len(translations) > 0 and key in translations:
            return str(translations[key])
        _LOGGER.warning(f"No translation found for key: {key}")
        return key

    def __repr__(self):
        return "<template Translated>"


class AllTranslations(TranslatableTemplate):

    def __init__(self, hass: HomeAssistant, available_languages):
        super().__init__(hass, available_languages)

    def __call__(self, language: str):
        self.validate_language(language)
        translations = {}
        translations.update(get_cached_translations(self._hass, language, "state"))
        translations.update(get_cached_translations(self._hass, language, "entity"))
        translations.update(get_cached_translations(self._hass, language, "entity_component"))
        return translations

    def __repr__(self):
        return "<template AllTranslations>"


class EvalTemplate:

    def __init__(self, hass: HomeAssistant):
        self._hass = hass

    def __call__(self, content: str):
        tpl = Template(content, self._hass)
        return tpl.async_render()

    def __repr__(self):
        return "<template EvalTemplate>"


def get_cached(
        self,
        language: str,
        category: str,
        components: set[str],
):
    cached = self.cache.get(language, {})
    return [cached.get(component, {}).get(category, {}) for component in components]


@bind_hass
async def load_translations_to_cache(
        hass: HomeAssistant,
        language: str,
):
    lock = hass.data.setdefault(TRANSLATION_LOAD_LOCK, asyncio.Lock())

    components_entities = {
        component for component in hass.config.components if "." not in component
    }
    components_state = set(hass.config.components)

    async with lock:
        cache = hass.data.setdefault(TRANSLATION_FLATTEN_CACHE, _TranslationCache(hass))
        await cache.async_fetch(language, "entity", components_entities)
        await cache.async_fetch(language, "states", components_state)
        await cache.async_fetch(language, "entity_component", components_state)


@bind_hass
def get_cached_translations(
        hass: HomeAssistant,
        language: str,
        category: str,
        integration=None,
):
    if integration is not None:
        components = {integration}
    elif category == "state":
        components = set(hass.config.components)
    else:
        components = {
            component for component in hass.config.components if "." not in component
        }

    cache = hass.data.setdefault(TRANSLATION_FLATTEN_CACHE, _TranslationCache(hass))
    # noinspection PyUnresolvedReferences
    cached = cache.ct_patched_get_cached(language, category, components)

    return dict(ChainMap(*cached))


# noinspection PyProtectedMember
def setup(hass: HomeAssistant, config: ConfigType):
    if DOMAIN not in config:
        return True
    languages = []
    if CONF_PRELOAD_TRANSLATIONS in config[DOMAIN]:
        languages = config[DOMAIN][CONF_PRELOAD_TRANSLATIONS]

        async def load_translations(_event: Event):
            for language in languages:
                await load_translations_to_cache(hass, language)

        hass.bus.async_listen(EVENT_COMPONENT_LOADED, load_translations)

    state_translated_template = StateTranslated(hass, languages)
    state_attr_translated_template = StateAttrTranslated(hass, languages)
    translated_template = Translated(hass, languages)
    all_translations_template = AllTranslations(hass, languages)
    eval_template = EvalTemplate(hass)
    is_available = IsAvailable(hass)
    convert_colors = ConvertColorSystem(hass)

    _TranslationCache.ct_patched_get_cached = get_cached

    def is_safe_callable(self: TemplateEnvironment, obj):
        # noinspection PyUnresolvedReferences
        return (isinstance(obj, (StateTranslated, StateAttrTranslated, EvalTemplate, Translated, AllTranslations, IsAvailable))
                or self.ct_original_is_safe_callable(obj))

    def patch_environment(env: TemplateEnvironment):
        env.globals[CONST_STATE_TRANSLATED_FUNCTION_NAME] = state_translated_template
        env.globals[CONST_STATE_ATTR_TRANSLATED_FUNCTION_NAME] = state_attr_translated_template
        env.globals[CONST_TRANSLATED_FUNCTION_NAME] = translated_template
        env.globals[CONST_ALL_TRANSLATIONS_FUNCTION_NAME] = all_translations_template
        env.globals[CONST_EVAL_FUNCTION_NAME] = eval_template
        env.globals[CONST_IS_AVAILABLE_FUNCTION_NAME] = is_available
        env.globals[CONST_COLOR_CONVERSION_NAME] = convert_colors
        env.filters[CONST_STATE_TRANSLATED_FUNCTION_NAME] = state_translated_template
        env.filters[CONST_STATE_ATTR_TRANSLATED_FUNCTION_NAME] = state_attr_translated_template
        env.filters[CONST_TRANSLATED_FUNCTION_NAME] = translated_template
        env.filters[CONST_EVAL_FUNCTION_NAME] = eval_template
        env.filters[CONST_IS_AVAILABLE_FUNCTION_NAME] = is_available
        env.filters[CONST_COLOR_CONVERSION_NAME] = convert_colors

    def patched_init(
        self: TemplateEnvironment,
        hass_param: HomeAssistant | None,
        limited: bool | None = False,
        strict: bool | None = False,
        log_fn: Callable[[int, str], None] | None = None,
    ):
        # noinspection PyUnresolvedReferences
        self.ct_original__init__(hass_param, limited, strict, log_fn)
        patch_environment(self)

    if not hasattr(TemplateEnvironment, 'ct_original__init__'):
        TemplateEnvironment.ct_original__init__ = TemplateEnvironment.__init__
        TemplateEnvironment.__init__ = patched_init

    if not hasattr(TemplateEnvironment, 'ct_original_is_safe_callable'):
        TemplateEnvironment.ct_original_is_safe_callable = TemplateEnvironment.is_safe_callable
        TemplateEnvironment.is_safe_callable = is_safe_callable

    tpl = Template("", hass)
    tpl._strict = False
    tpl._limited = False
    patch_environment(tpl._env)
    tpl._strict = True
    tpl._limited = False
    patch_environment(tpl._env)

    return True