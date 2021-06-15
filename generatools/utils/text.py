def str_to_digit(s: str) -> int:
    """Transform str to digits, raise an exception if cannot"""
    if not s.isdigit():
        raise ValueError("s does not contain digits only: {}".format(s))
    else:
        i = int(s)
        return i
