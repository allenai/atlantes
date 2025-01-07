

Generate grpc python code from protobuf

```
python -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. ./changepoint.proto
```
