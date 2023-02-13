# AttrValue
AttrValue是一个记录了op可能的数据格式
```proto
message AttrValue {
  // LINT.IfChange
  message ListValue {
    repeated bytes s = 2;                        // "list(string)"
    repeated int64 i = 3 [packed = true];        // "list(int)"
    repeated float f = 4 [packed = true];        // "list(float)"
    repeated bool b = 5 [packed = true];         // "list(bool)"
    repeated DataType type = 6 [packed = true];  // "list(type)"
    repeated TensorShapeProto shape = 7;         // "list(shape)"
    repeated TensorProto tensor = 8;             // "list(tensor)"
    repeated NameAttrList func = 9;              // "list(attr)"
  }
  // LINT.ThenChange(//tensorflow/c/c_api.cc)

  oneof value {
    bytes s = 2;                 // "string"
    int64 i = 3;                 // "int"
    float f = 4;                 // "float"
    bool b = 5;                  // "bool"
    DataType type = 6;           // "type"
    TensorShapeProto shape = 7;  // "shape"
    TensorProto tensor = 8;      // "tensor"
    ListValue list = 1;          // any "list(...)"

    // "func" represents a function. func.name is a function's name or
    // a primitive op's name. func.attr.first is the name of an attr
    // defined for that function. func.attr.second is the value for
    // that attr in the instantiation.
    NameAttrList func = 10;

    // This is a placeholder only used in nodes defined inside a
    // function.  It indicates the attr value will be supplied when
    // the function is instantiated.  For example, let us suppose a
    // node "N" in function "FN". "N" has an attr "A" with value
    // placeholder = "foo". When FN is instantiated with attr "foo"
    // set to "bar", the instantiated node N's attr A will have been
    // given the value "bar".
    string placeholder = 9;
  }
}
```


# AttrSlice

AttrSlice
```cpp
class AttrSlice {
 public:
  AttrSlice(const NodeDef& node_def);  // NOLINT(runtime/explicit)

  AttrSlice();  // Empty
  explicit AttrSlice(const AttrValueMap* a);

  int size() const { return attrs()->size(); }

  // Returns the attr with attr_name if found.  Otherwise, returns
  // nullptr.
  const AttrValue* Find(StringPiece attr_name) const;
  const AttrValue* FindByString(const std::string& attr_name) const;

  // Returns the attr_value for attr_name if found. Otherwise, returns a
  // NotFound status.
  Status Find(StringPiece attr_name, const AttrValue** attr_value) const;
  Status FindByString(const std::string& attr_name,
                      const AttrValue** attr_value) const;

  // Helper class to avoid allocations in EqualAttrs.
  // TODO(irving): Will go away once NodeInfo is used.
  struct Scratch {
    std::string a;
    std::string b;
  };
}
  ```

# NameRangeMap

```cpp
typedef gtl::FlatMap<StringPiece, std::pair<int, int>, hash<StringPiece>>
    NameRangeMap;
```



# StringPiece
一个string_view类型的变量可以被想象成一个“镜像”，映射了一段已经存在的字符列表。更明确地说，一个string_view仅仅包含一个指针和一个长度，用以定位一个字符数据区间。string_view既不拥有这些数据，又不能修改这段存储。因此，复制string_view是浅拷贝，字符串内容不会被复制。