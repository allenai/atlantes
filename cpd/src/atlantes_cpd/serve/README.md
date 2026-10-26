

Generate grpc python code from protobuf

```
cd cpd/src
pip install grpcio-tools  # not in the dev extras: it requires a newer protobuf than the runtime pin
python -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. atlantes_cpd/serve/changepoint.proto
```
