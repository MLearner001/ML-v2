import re

def _parse_mcall(line: str):
    """Mengekstrak definisi siklus modal bor MCALL CYCLE81/82/83."""
    clean_line = line.strip().upper()
    print(f"Parsing: {clean_line}")
    if "MCALL" in clean_line:
      if "CYCLE" in clean_line:
        match = re.search(r"CYCLE\d+\s*\(([^)]*)\)", clean_line)
        if match:
          raw_args = match.group(1).split(",")
          args = [
              float(arg.strip()) if arg.strip() else None
              for arg in raw_args
          ]
          args.extend([None] * (6 - len(args)))

          rtp = args[0] if args[0] is not None else 0.0
          rfp = args[1] if args[1] is not None else 0.0
          sdis = args[2] if args[2] is not None else 0.0
          dp = args[3]
          dpr = args[4]

          if dp is None and dpr is not None:
             dp_val = rfp - abs(dpr)
          elif dp is not None:
             dp_val = dp
          else:
             dp_val = 0.0

          print(f"Parsed args: {args}")
          print(f"Params: RTP={rtp}, RFP={rfp}, SDIS={sdis}, DP={dp_val}")
      else:
        print("MCALL off")

# CYCLE82(RTP, RFP, SDIS, DP, DPR, DTB)
_parse_mcall("MCALL CYCLE82(100, 50, 5, -20, , 1)")
_parse_mcall("MCALL CYCLE82(100, 50, 5, , 20, 1)")
_parse_mcall("MCALL CYCLE82(100, 50, 5, , , 1)")
_parse_mcall("MCALL")
