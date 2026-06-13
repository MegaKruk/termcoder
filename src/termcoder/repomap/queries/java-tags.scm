(class_declaration
  name: (identifier) @name.definition.class) @definition.class

(interface_declaration
  name: (identifier) @name.definition.interface) @definition.interface

(method_declaration
  name: (identifier) @name.definition.method) @definition.method

(method_invocation
  name: (identifier) @name.reference.call) @reference.call

(object_creation_expression
  type: (type_identifier) @name.reference.class) @reference.class
