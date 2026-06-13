(function_item
  name: (identifier) @name.definition.function) @definition.function

(struct_item
  name: (type_identifier) @name.definition.class) @definition.class

(enum_item
  name: (type_identifier) @name.definition.class) @definition.class

(trait_item
  name: (type_identifier) @name.definition.interface) @definition.interface

(mod_item
  name: (identifier) @name.definition.module) @definition.module

(call_expression
  function: (identifier) @name.reference.call) @reference.call

(call_expression
  function: (field_expression
    field: (field_identifier) @name.reference.call)) @reference.call

(call_expression
  function: (scoped_identifier
    name: (identifier) @name.reference.call)) @reference.call
