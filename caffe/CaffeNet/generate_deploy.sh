awk '
/^layer {/ { in_layer = 1; }
in_layer == 1 { layer = layer "\n" $0; }
in_layer == 1 && $1 == "type:" { type = $2 }
/^}/ {
  if (type =="\"SoftmaxWithLoss\"") {
    layer = "layer {\n  name: \"prob\"\n  type: \"Softmax\"\n  bottom: \"fc8-cats-dogs\"\n  top: \"prob\"\n}"
  }
  if(type != "\"Data\"" && type != "\"Accuracy\""){print layer}
  layer = ""; type = ""; in_layer = 0; next
}
in_layer == 0 { print }
' < train_val.prototxt > deploy.prototxt
