from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from mvi_v3.data.events import WindowRecord


def reconstruct_center_priority(
    windows: Sequence[WindowRecord],
    predictions: Sequence[np.ndarray],
) -> dict[str, list[float]]:
    best: dict[str, dict[int, tuple[float, int, float]]] = defaultdict(dict)
    for window_order, (window, pred) in enumerate(zip(windows, predictions, strict=True)):
        center = (len(pred) - 1) / 2.0
        for local_index, note_index in enumerate(window.global_note_indices.tolist()):
            if note_index < 0 or window.padding_mask[local_index]:
                continue
            score = (abs(local_index - center), window_order)
            current = best[window.piece_id].get(note_index)
            if current is None or score < current[:2]:
                best[window.piece_id][note_index] = (score[0], score[1], float(pred[local_index]))

    return {
        piece_id: [per_note[note_index][2] for note_index in sorted(per_note)]
        for piece_id, per_note in best.items()
    }
