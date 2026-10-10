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
    # requires_match: per bucket (keyed by id), its compound rules for the
    # fast matcher and its relationship rules as a SoupSieve selector list,
    # so an element meets only the rules it can match
    lists: dict[int, tuple[list[Any], Any]] | None = None


def _bucket_lists(plan: _Plan) -> dict[int, tuple[list[Any], Any]] | None:
    """Split each bucket into compound rules and a SoupSieve list of the rest."""
    try:
        from soupsieve import css_types

        top = plan.compiled.selectors
        lists: dict[int, tuple[list[Any], Any]] = {}
        for bucket in (plan.by_tag, plan.by_id, plan.by_class, plan.by_attribute):
            for rules in bucket.values():
                simple = [rule for rule in rules if _supported(rule)]
                related = [rule for rule in rules if not _supported(rule)]
                lists[id(rules)] = (
                    simple,
                    css_types.SelectorList(related, top.is_not, top.is_html)
                    if related
                    else None,
                )
        return lists
    except (AttributeError, TypeError, ImportError):
        return None


def _matcher(plan: _Plan, root: Tag) -> Any:
    """One SoupSieve matcher for a whole select(), as SoupSieve's own select().

    ``compiled.match(el)`` builds a matcher per element, each walking up to the
    document root first. The plan admits no :scope, :root or other state that
    depends on the matcher's scope, so one built for *root* matches the same.
    """
    if plan.lists is None:
        return None
    try:
        from soupsieve.css_match import CSSMatch

        compiled = plan.compiled
        return CSSMatch(compiled.selectors, root, compiled.namespaces, compiled.flags)
    except (AttributeError, TypeError, ImportError):
        return None


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
        if requires_match:
            plan.lists = _bucket_lists(plan)
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
    matcher = _matcher(plan, root) if plan.requires_match else None
    lists = plan.lists if matcher is not None else None
    requires_match = plan.requires_match
    universal = plan.universal
    by_tag, by_id = plan.by_tag, plan.by_id
    by_class, by_attribute = plan.by_class, plan.by_attribute
    for el in root.descendants:
        if not isinstance(el, Tag):
            continue
        attrs = el.attrs
        # Parsed HTML has normalized strings/string lists. A caller may mutate
        # attrs to arbitrary Python values; retain SoupSieve's normalization.
        if attrs and not _plain_attrs(attrs):
            found = plan.compiled.match(el)
        else:
            name = el.name.lower()
            # Only the buckets this element can match; most elements have none
            groups = [universal] if universal else []
            group = by_tag.get(name)
            if group:
                groups.append(group)
            classes: list[str] = []
            if attrs:
                if by_id:
                    group = by_id.get(cast(str, attrs.get("id", "")))
                    if group:
                        groups.append(group)
                value = attrs.get("class")
                if value:
                    classes = value.split() if isinstance(value, str) else value
                if by_class:
                    for value in classes:
                        group = by_class.get(value)
                        if group:
                            groups.append(group)
                if by_attribute:
                    for key in attrs:
                        group = by_attribute.get(key)
                        if group:
                            groups.append(group)
            if not groups:
                continue
            if requires_match:
                # Keep ancestor/sibling and negation semantics in SoupSieve.
                # Scope-dependent states were rejected while building the plan.
                if lists is None or matcher is None:
                    found = plan.compiled.match(el)
                else:
                    found = False
                    for group in groups:
                        simple, related = lists[id(group)]
                        if any(
                            _matches(rule, name, attrs, classes) for rule in simple
                        ) or (
                            related is not None and matcher.match_selectors(el, related)
                        ):
                            found = True
                            break
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


def _plain_attrs(attrs: dict) -> bool:
    """Whether attrs are as a parser leaves them: lowercase string keys,
    string, None or list-of-string values (and a string id).

    Run for every element of every select(), so plain ``str`` keys and values
    take the shortest path; anything unusual (a ``str`` subclass included)
    answers "not plain", which only sends the element to SoupSieve.
    """
    for key, value in attrs.items():
        if key.__class__ is not str or not (key.islower() or key == key.lower()):
            return False
        if value.__class__ is str:
            continue
        if value is None:
            if key == "id":
                return False
            continue
        if isinstance(value, str):
            continue
        if key == "id" or not isinstance(value, list):
            return False
        for item in value:
            if not isinstance(item, str):
                return False
    return True


def matches(el: Tag, selector: str) -> bool:
    """``el.css.match(selector)``, through the indexed plan when it applies.

    SoupSieve builds a matcher per call (walking up to the document root)
    and tries every rule; most elements have no rule they could match.
    """
    if el._is_xml:
        return el.css.match(selector)
    plan = _plan(selector)
    if plan is None:
        return el.css.match(selector)
    attrs = el.attrs
    if attrs and not _plain_attrs(attrs):
        return el.css.match(selector)
    name = el.name.lower()
    groups = [plan.universal] if plan.universal else []
    group = plan.by_tag.get(name)
    if group:
        groups.append(group)
    classes: list[str] = []
    if attrs:
        if plan.by_id:
            group = plan.by_id.get(cast(str, attrs.get("id", "")))
            if group:
                groups.append(group)
        value = attrs.get("class")
        if value:
            classes = value.split() if isinstance(value, str) else value
        if plan.by_class:
            for value in classes:
                group = plan.by_class.get(value)
                if group:
                    groups.append(group)
        if plan.by_attribute:
            for key in attrs:
                group = plan.by_attribute.get(key)
                if group:
                    groups.append(group)
    if not groups:
        return False
    if not plan.requires_match:
        return any(
            _matches(rule, name, attrs, classes) for group in groups for rule in group
        )
    matcher = _matcher(plan, el)
    if matcher is None or plan.lists is None:
        return el.css.match(selector)
    for group in groups:
        simple, related = plan.lists[id(group)]
        if any(_matches(rule, name, attrs, classes) for rule in simple):
            return True
        if related is not None and matcher.match_selectors(el, related):
            return True
    return False


def closest(el: Tag, selector: str) -> Tag | None:
    """``el.css.closest(selector)``: *el* or its nearest matching ancestor."""
    if el._is_xml or _plan(selector) is None:
        return el.css.closest(selector)
    current: Any = el
    while current is not None:
        if matches(current, selector):
            return current
        current = current.parent
    return None


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
