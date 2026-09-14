"""Execute simple HTML selector groups in one indexed pass.

SoupSieve remains the parser and the authority for CSS semantics. Its compiled
attribute regexes already encode escaping, operators and case flags. We only
accelerate compound selectors and their simple :not/:is groups. Relationship
selectors use the same index to narrow candidates, then the original engine to
match them. Namespaces, pseudo states and XML keep the original tree scan. Plans
contain no DOM nodes, so mutation and concurrent extractions cannot leave stale
node caches.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import soupsieve
from bs4 import Tag


def _supported(rule: Any, *, relationships: bool = False) -> bool:
    if not hasattr(rule, "tag"):
        return False
    if rule.flags or rule.nth or rule.contains or rule.lang:
        return False
    if rule.relation and not relationships:
        return False
    if rule.rel_type not in (None, " ", ">", "+", "~"):
        return False
    if rule.tag is not None and rule.tag.prefix is not None:
        return False
    if any(a.prefix not in (None, "") for a in rule.attributes):
        return False
    return all(
        not group.is_html
        and all(_supported(child, relationships=relationships) for child in group)
        for group in (*rule.selectors, rule.relation)
    )


def _matches(rule: Any, name: str, attrs: dict, classes: list[str]) -> bool:
    if rule.tag and rule.tag.name not in (None, "*") and rule.tag.name.lower() != name:
        return False
    if rule.ids and any(attrs.get("id") != value for value in rule.ids):
        return False
    if rule.classes and any(value not in classes for value in rule.classes):
        return False
    for attribute in rule.attributes:
        key = attribute.attribute.lower()
        if key not in attrs:
            return False
        if attribute.pattern is not None:
            value = attrs[key]
            if isinstance(value, list):
                value = " ".join(value)
            elif value is None:
                value = ""
            if attribute.pattern.match(value) is None:
                return False
    for group in rule.selectors:
        found = any(_matches(child, name, attrs, classes) for child in group)
        if found == group.is_not:
            return False
    return True


@dataclass
class _Plan:
    compiled: Any
    by_tag: dict[str, list[Any]]
    by_id: dict[str, list[Any]]
    by_class: dict[str, list[Any]]
    by_attribute: dict[str, list[Any]]
    universal: list[Any]
    requires_match: bool = False


@lru_cache(maxsize=256)
def _plan(selector: str) -> _Plan | None:
    compiled = soupsieve.compile(selector)
    try:
        requires_match = not all(_supported(rule) for rule in compiled.selectors)
        if requires_match and not all(
            _supported(rule, relationships=True) for rule in compiled.selectors
        ):
            return None
        plan = _Plan(
            compiled,
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            [],
            requires_match=requires_match,
        )
        for rule in cast(Any, compiled.selectors):
            if rule.ids:
                plan.by_id[rule.ids[0]].append(rule)
            elif rule.classes:
                plan.by_class[rule.classes[0]].append(rule)
            elif rule.tag and rule.tag.name not in (None, "*"):
                plan.by_tag[rule.tag.name.lower()].append(rule)
            elif rule.attributes:
                plan.by_attribute[rule.attributes[0].attribute.lower()].append(rule)
            else:
                plan.universal.append(rule)
        if requires_match and plan.universal:
            # Without an anchor every node would need the reference matcher;
            # its normal tree scan is more efficient in that case.
            return None
        return plan
    except AttributeError:
        # Future SoupSieve AST revisions must fall back, never guess semantics.
        return None


def select(root: Tag, selector: str, *, limit: int = 0) -> list[Tag]:
    """Return CSS matches in document order, respecting the current live DOM."""
    if root._is_xml:
        return root.select(selector, limit=limit)
    plan = _plan(selector)
    if plan is None:
        return root.select(selector, limit=limit)
    if (
        not plan.requires_match
        and len(plan.by_tag) == 1
        and not (plan.by_id or plan.by_class or plan.by_attribute or plan.universal)
    ):
        rules = next(iter(plan.by_tag.values()))
        if all(not (r.ids or r.classes or r.attributes or r.selectors) for r in rules):
            name = next(iter(plan.by_tag))
            result = []
            for el in root.descendants:
                if isinstance(el, Tag) and el.name.lower() == name:
                    result.append(el)
                    if limit > 0 and len(result) >= limit:
                        break
            return result
    matches = []
    for el in root.descendants:
        if not isinstance(el, Tag):
            continue
        attrs = el.attrs
        # Parsed HTML has normalized strings/string lists. A caller may mutate
        # attrs to arbitrary Python values; retain SoupSieve's normalization.
        if ("id" in attrs and not isinstance(attrs["id"], str)) or any(
            not isinstance(key, str)
            or key != key.lower()
            or not (
                value is None
                or isinstance(value, str)
                or isinstance(value, list)
                and all(isinstance(v, str) for v in value)
            )
            for key, value in attrs.items()
        ):
            found = plan.compiled.match(el)
        else:
            name = el.name.lower()
            classes = attrs.get("class", [])
            if isinstance(classes, str):
                classes = classes.split()
            if classes is None:
                classes = []
            groups = [
                plan.universal,
                plan.by_tag.get(name, ()),
                plan.by_id.get(cast(str, attrs.get("id", "")), ()),
            ]
            groups.extend(
                plan.by_class[value] for value in classes if value in plan.by_class
            )
            groups.extend(
                plan.by_attribute[key] for key in attrs if key in plan.by_attribute
            )
            if plan.requires_match:
                # Keep ancestor/sibling and negation semantics in SoupSieve.
                # Scope-dependent states were rejected while building the plan.
                found = any(groups) and plan.compiled.match(el)
            else:
                found = any(
                    _matches(rule, name, attrs, classes)
                    for group in groups
                    for rule in group
                )
        if found:
            matches.append(el)
            if limit > 0 and len(matches) >= limit:
                break
    return matches


def select_one(root: Tag, selector: str) -> Tag | None:
    matches = select(root, selector, limit=1)
    return matches[0] if matches else None


@dataclass
class _ManyPlan:
    plans: tuple[_Plan, ...]
    by_tag: dict[str, list[tuple[int, Any]]]
    by_id: dict[str, list[tuple[int, Any]]]
    by_class: dict[str, list[tuple[int, Any]]]
    by_attribute: dict[str, list[tuple[int, Any]]]
    universal: list[tuple[int, Any]]


@lru_cache(maxsize=64)
def _many_plan(selectors: tuple[str, ...]) -> _ManyPlan | None:
    plans = tuple(_plan(selector) for selector in selectors)
    if any(plan is None or plan.requires_match for plan in plans):
        return None
    result = _ManyPlan(
        cast(tuple[_Plan, ...], plans),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        [],
    )
    for index, plan in enumerate(result.plans):
        for source, target in (
            (plan.by_tag, result.by_tag),
            (plan.by_id, result.by_id),
            (plan.by_class, result.by_class),
            (plan.by_attribute, result.by_attribute),
        ):
            for key, rules in source.items():
                target[key].extend((index, rule) for rule in rules)
        result.universal.extend((index, rule) for rule in plan.universal)
    return result


def select_many(root: Tag, selectors: tuple[str, ...]) -> list[list[Tag]]:
    """Run independent simple queries in one walk, retaining each query's order.

    Cached plans contain only selector rules. Matches and node identities are
    local to this call, so tree and attribute mutations remain visible.
    """
    if not selectors:
        return []
    plan = None if root._is_xml else _many_plan(selectors)
    if plan is None:
        return [select(root, selector) for selector in selectors]
    results: list[list[Tag]] = [[] for _ in selectors]
    for el in root.descendants:
        if not isinstance(el, Tag):
            continue
        attrs = el.attrs
        if ("id" in attrs and not isinstance(attrs["id"], str)) or any(
            not isinstance(key, str)
            or key != key.lower()
            or not (
                value is None
                or isinstance(value, str)
                or isinstance(value, list)
                and all(isinstance(v, str) for v in value)
            )
            for key, value in attrs.items()
        ):
            for index, query in enumerate(plan.plans):
                if query.compiled.match(el):
                    results[index].append(el)
            continue
        name = el.name.lower()
        classes = attrs.get("class", [])
        if isinstance(classes, str):
            classes = classes.split()
        if classes is None:
            classes = []
        groups = [
            plan.universal,
            plan.by_tag.get(name, ()),
            plan.by_id.get(cast(str, attrs.get("id", "")), ()),
        ]
        groups.extend(
            plan.by_class[value] for value in classes if value in plan.by_class
        )
        groups.extend(
            plan.by_attribute[key] for key in attrs if key in plan.by_attribute
        )
        matched: set[int] = set()
        for group in groups:
            for index, rule in group:
                if index not in matched and _matches(rule, name, attrs, classes):
                    matched.add(index)
                    results[index].append(el)
    return results
