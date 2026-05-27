from dotenv import load_dotenv
from langfuse import get_client
from opentelemetry import trace as otel_trace

load_dotenv()

langfuse = get_client()

provider = otel_trace.get_tracer_provider()
print(f"TracerProvider type: {type(provider).__name__}")
print(f"TracerProvider module: {type(provider).__module__}")

# Проверка: процессоры (куда отправляются spans)
if hasattr(provider, "_active_span_processor"):
    processor = provider._active_span_processor
    print(f"\nActive span processor: {type(processor).__name__}")
    if hasattr(processor, "_span_processors"):
        for i, p in enumerate(processor._span_processors):
            print(f"  [{i}] {type(p).__name__}")
