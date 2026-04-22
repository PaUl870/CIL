import numpy as np
from collections import defaultdict

def create_coldstart_split(train_df, valid_df, wishlist_df, cold_ratio=0.10, seed=42):
    """
    Creates cold-start entities by removing their purchase edges from train_df only.
    Val edges for cold entities are KEPT as cold-start evaluation targets.
    Cold entities retain wishlist edges as their only training signal.

    Returns:
        new_train_df, cold_users, cold_items
    """
    rng = np.random.default_rng(seed)

    wishlist_users = set(wishlist_df["sid"].values)
    wishlist_items = set(wishlist_df["pid"].values)

    val_sids = valid_df["sid"].values
    val_pids = valid_df["pid"].values

    n_val_cold = int(len(val_sids) * cold_ratio)

    def pick_cold_entities(val_ids, wishlist_set, n_target):
        counts = defaultdict(int)
        for eid in val_ids:
            if eid in wishlist_set:
                counts[eid] += 1
        candidates = list(counts.keys())
        rng.shuffle(candidates)
        selected, total = set(), 0
        for c in sorted(candidates, key=lambda x: -counts[x]):
            if total >= n_target:
                break
            selected.add(c)
            total += counts[c]
        return selected

    cold_users = pick_cold_entities(val_sids, wishlist_users, n_val_cold)
    cold_items = pick_cold_entities(val_pids, wishlist_items, n_val_cold)

    print(f"Selected {len(cold_users)} cold users, {len(cold_items)} cold items")

    # Remove cold entity purchase edges from train only
    train_mask = ~(
        train_df["sid"].isin(cold_users) | train_df["pid"].isin(cold_items)
    )
    new_train_df = train_df[train_mask].reset_index(drop=True)
    print(f"Removed {(~train_mask).sum()} purchase edges from train ({(~train_mask).mean():.1%})")

    # Verify cold-start ratio in val (edges kept, just checking coverage)
    val_cold = valid_df["sid"].isin(cold_users) | valid_df["pid"].isin(cold_items)
    print(f"Val cold-start edges: {val_cold.sum()}/{len(valid_df)} ({val_cold.mean():.1%})")

    # Sanity checks
    assert not new_train_df["sid"].isin(cold_users).any(), "Cold users still in train!"
    assert not new_train_df["pid"].isin(cold_items).any(), "Cold items still in train!"
    print(f"Cold users with wishlist signal: {len(cold_users & wishlist_users)}/{len(cold_users)}")
    print(f"Cold items with wishlist signal: {len(cold_items & wishlist_items)}/{len(cold_items)}")

    return new_train_df, cold_users, cold_items