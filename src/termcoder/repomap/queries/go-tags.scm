(function_declaration
  name: (identifier) @name.definition.function) @definition.function

(method_declaration
  name: (field_identifier) @name.definition.method) @definition.method

(type_declaration
  (type_spec name: (type_identifier) @name.definition.type)) @definition.type

(call_expression
  function: (identifier) @name.reference.call) @reference.call

(call_expression
  function: (selector_expression
    field: (field_identifier) @name.reference.call)) @reference.call
