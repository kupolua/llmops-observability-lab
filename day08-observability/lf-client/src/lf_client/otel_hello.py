from dotenv import load_dotenv
from langfuse import get_client
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

load_dotenv()


def main() -> None:
    langfuse = get_client()  # noqa: F841

    tracer = otel_trace.get_tracer("my-service")

    with tracer.start_as_current_span("hello-otel") as span:
        # ВАЖНО: явно говорим Langfuse, что это «span»-наблюдение
        span.set_attribute("langfuse.observation.type", "span")
        span.set_attribute("langfuse.internal.as_root", True)

        span.set_attribute("service.version", "0.1.0")
        span.set_attribute("my.custom.field", "hello world")

        span.add_event("processing-started", {"phase": "init"})

        import time

        time.sleep(0.1)

        span.add_event("processing-finished")
        span.set_status(Status(StatusCode.OK))

    langfuse.flush()
    print("Span отправлен. Открой Langfuse → Traces → найди 'hello-otel'")


if __name__ == "__main__":
    main()
