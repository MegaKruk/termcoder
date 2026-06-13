(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name.definition.function)) @definition.function

(class_specifier
  name: (type_identifier) @name.definition.class) @definition.class

(struct_specifier
  name: (type_identifier) @name.definition.class) @definition.class

(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @name.definition.method))) @definition.method

(call_expression
  function: (identifier) @name.reference.call) @reference.call

(call_expression
  function: (field_expression
    field: (field_identifier) @name.reference.call)) @reference.call
