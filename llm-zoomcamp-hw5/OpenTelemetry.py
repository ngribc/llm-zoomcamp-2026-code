from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

with tracer.start_as_current_span("my_operation") as span:
    # your code here
    span.set_attribute("my_key", "my_value")