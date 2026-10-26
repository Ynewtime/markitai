// The API address as a combobox: an address field that, focused or clicked, lists the provider's
// known addresses under it (a listbox, not `<datalist>`, which Safari shows unreliably). Picking one
// fills the field; anything typed stays as typed and the list keeps every address. The arrow keys
// walk the list, Enter picks, Escape and a tap elsewhere close it.
import { useEffect, useRef, useState } from "preact/hooks";
import { type EndpointChoice, matchingChoice } from "../lib/endpoints.ts";
import { Icon } from "./icons.tsx";

export function AddressField({
  id,
  value,
  choices,
  label,
  listLabel,
  placeholder,
  disabled,
  describedBy,
  onInput,
}: {
  id: string;
  value: string;
  choices: readonly EndpointChoice[];
  /** A choice's name ("International · Coding Plan"). */
  label: (choice: EndpointChoice) => string;
  /** The list's accessible name. */
  listLabel: string;
  placeholder?: string;
  disabled?: boolean;
  describedBy?: string;
  onInput: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const root = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const shown = open && choices.length > 0 && !disabled;
  const current = matchingChoice(choices, value);
  const listId = `${id}-list`;

  // A tap or click outside closes the list (iOS keeps the field focused when the page is tapped).
  useEffect(() => {
    if (!shown) return;
    const away = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", away, true);
    return () => document.removeEventListener("pointerdown", away, true);
  }, [shown]);
  // The active entry stays in view while the arrows walk a list that scrolls.
  useEffect(() => {
    if (shown && active >= 0) list.current?.children[active]?.scrollIntoView?.({ block: "nearest" });
  }, [shown, active]);

  const show = () => {
    if (open) return;
    setOpen(true);
    setActive(-1);
  };
  const pick = (choice: EndpointChoice) => {
    onInput(choice.base);
    setOpen(false);
    setActive(-1);
    input.current?.focus();
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (choices.length === 0 || disabled) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const down = event.key === "ArrowDown";
      if (!open) {
        setOpen(true);
        const at = current ? choices.indexOf(current) : -1;
        setActive(at >= 0 ? at : down ? 0 : choices.length - 1);
        return;
      }
      setActive((index) => (index < 0 ? (down ? 0 : choices.length - 1) : (index + (down ? 1 : choices.length - 1)) % choices.length));
    } else if (event.key === "Enter" && shown && active >= 0) {
      // Picks the entry instead of submitting the form.
      event.preventDefault();
      pick(choices[active]!);
    } else if (event.key === "Escape" && shown) {
      // Closes the list only, not a dialog around it.
      event.preventDefault();
      event.stopPropagation();
      setOpen(false);
      setActive(-1);
    } else if (event.key === "Tab") setOpen(false);
  };

  return (
    <div ref={root} class="addr-combo" onFocusOut={(event) => !root.current?.contains(event.relatedTarget as Node | null) && setOpen(false)}>
      <input
        ref={input}
        id={id}
        type="url"
        inputMode="url"
        role="combobox"
        // No <datalist>: the list below is the page's own.
        list={undefined}
        aria-autocomplete="none"
        aria-expanded={shown}
        aria-controls={choices.length > 0 ? listId : undefined}
        aria-activedescendant={shown && active >= 0 ? `${listId}-${active}` : undefined}
        aria-describedby={describedBy}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        autoComplete="off"
        autoCapitalize="off"
        spellcheck={false}
        onFocus={show}
        onClick={show}
        onKeyDown={onKeyDown}
        onInput={(event) => {
          onInput(event.currentTarget.value);
          setActive(-1);
          setOpen(true);
        }}
      />
      {choices.length > 0 && !disabled && (
        <span class="addr-combo-caret" aria-hidden="true">
          <Icon name="CaretDown" size={11} />
        </span>
      )}
      {shown && (
        <ul ref={list} id={listId} class="addr-combo-list" role="listbox" aria-label={listLabel}>
          {choices.map((choice, index) => (
            <li
              key={choice.base}
              id={`${listId}-${index}`}
              role="option"
              aria-selected={choice === current}
              class={index === active ? "is-active" : undefined}
              title={choice.base}
              // The field keeps the focus (and iOS its keyboard) while an entry is tapped.
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => pick(choice)}
            >
              <span class="addr-combo-label">{label(choice)}</span>
              <span class="addr-combo-base">{choice.base}</span>
              <span class="addr-combo-check" aria-hidden="true">
                {choice === current && <Icon name="Check" size={13} />}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
