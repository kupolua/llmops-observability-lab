from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

lf = get_client()

print("=== Langfuse client methods ===")
for m in dir(lf):
    if not m.startswith("_"):
        print(f"  {m}")

with lf.start_as_current_observation(as_type="span", name="test") as span:
    print("\n=== Span methods ===")
    for m in dir(span):
        if not m.startswith("_"):
            print(f"  {m}")
