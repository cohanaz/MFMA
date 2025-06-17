import streamlit as st

def stqdm(iterable, total=None, desc=None):
    if total is None:
        total = len(iterable)
    
    desc_col, progress_col = st.columns([2, 3])
    with desc_col:
        st.markdown(desc + f" ({total} items):")
    with progress_col:
        progress = st.progress(0)
        for i, item in enumerate(iterable):
            yield item
            progress.progress(int((i + 1) / total * 100))

def align_dicts_by_keys(dict1, dict2):
    """
    Verify both dicts have identical keys and return them ordered by dict1's key order.
    
    Returns:
        aligned_dict1, aligned_dict2
    Raises:
        ValueError if the dicts don't share identical keys
    """
    if set(dict1.keys()) != set(dict2.keys()):
        missing_1 = set(dict2.keys()) - set(dict1.keys())
        missing_2 = set(dict1.keys()) - set(dict2.keys())
        raise ValueError(f"Key mismatch:\nOnly in dict1: {missing_2}\nOnly in dict2: {missing_1}")

    # Order both dicts by dict1's key order
    ordered_keys = list(dict1.keys())
    dict1_ordered = {k: dict1[k] for k in ordered_keys}
    dict2_ordered = {k: dict2[k] for k in ordered_keys}
    
    return dict1_ordered, dict2_ordered
