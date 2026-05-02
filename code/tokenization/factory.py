def create_tokenizer(name: str, **kwargs):
    """Factory function to create a tokenizer by name."""
    normalized = "wl" if name == "graphs" else name

    if normalized == "wl":
        # Import inside to avoid hard dependency if wlplan is missing
        # and user is using other tokenizers
        try:
            from code.tokenization.wl import WLTokenizer
            return WLTokenizer(iterations=kwargs.get("iterations", 2))
        except ImportError:
            raise ImportError("WLTokenizer requires 'wlplan' package.")
            
    elif normalized == "shortest_path":
        from code.tokenization.shortest_path import ShortestPathTokenizer

        return ShortestPathTokenizer(
            max_path_length=kwargs.get("max_path_length", 5),
        )
    else:
        raise ValueError(f"Tokenizer is not included in this downstream snapshot: {name}")
