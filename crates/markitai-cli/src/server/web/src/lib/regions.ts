// Providers whose keys belong to a region (Moonshot AI, Kimi Code, MiniMax, Z.ai, Model Studio): the
// server lists their endpoints on the card (`regions`, contract v1.25), tries a pasted key at each
// in turn when no address is typed, and names the one that listed the models (`api_base`), which
// the page fills in and saves. An older backend sends no `regions`: the field then shows the
// provider's default address, as before, also for a card whose models are entered by hand: with
// no model list there is nothing to try a key at.
import type { ProviderCard } from "../api/types.ts";

/** Whether the address of a key pasted on this card is found from the key. */
export const foundFromKey = (card: Partial<Pick<ProviderCard, "regions" | "supports_discovery">> | null | undefined): boolean =>
  card?.supports_discovery !== false && (card?.regions?.length ?? 0) > 1;
