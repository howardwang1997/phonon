"""Write TMD monolayer xyz files for td_anharmonic.py (T2 TDEP-MD screen).
Builds each CDW material via ase.build.mx2 and saves to data/lchannel/<name>.xyz."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ase.build import mx2
from ase.io import write
MATS = [
    ("NbSe2", "NbSe2", "2H", 3.44, 3.34),
    ("NbS2", "NbS2", "2H", 3.33, 3.00),
    ("2H-TaS2", "TaS2", "2H", 3.31, 3.00),
    ("2H-TaSe2", "TaSe2", "2H", 3.43, 3.30),
    ("1T-VSe2", "VSe2", "1T", 3.34, 3.00),
    ("1T-TiSe2", "TiSe2", "1T", 3.53, 2.90),
]
out = ROOT / "data" / "lchannel"; out.mkdir(parents=True, exist_ok=True)
for name, formula, poly, a, thick in MATS:
    kind = "2H" if poly == "2H" else "1T"
    at = mx2(formula=formula, kind=kind, a=a, thickness=thick, vacuum=7.5)
    at.pbc = True
    write(str(out / f"{name}.xyz"), at)
    print(f"wrote {name}.xyz ({len(at)} atoms, {poly})")
