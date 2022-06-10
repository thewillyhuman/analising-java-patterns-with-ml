SELECT programs.program_id as program__program_id,
       programs.classpercentage as programs__percentage_of_classes,
       programs.interfacepercentage as programs__percentage_of_interfaces,
       programs.enumpercentage as programs__percentage_of_enums,
       programs.codeinpackages as programs__contains_code_in_packages,
       programs.defaultpackage as programs__contains_code_in_default_package,

       type_defs.nodetype_id as type_def__node_id,
       type_defs.typenodetype as type_def__syntactic_category,
       type_defs.typepublicvisibility as type_def__visibility,
       type_defs.isfinalclass as type_def__is_final,
       type_defs.hasextends as type_def__is_extension,
       type_defs.numberannotations as type_def__number_of_annotations,
       type_defs.isindefaultpackage as type_def__is_in_default_package,
       type_defs.numberimplements as type_def__number_of_implements,
       type_defs.numbergenerictypes as type_def__number_of_generics,
       type_defs.numbermethods as type_def__number_of_methods,
       type_defs.percentageoverloadedmethods as type_def__percentage_overloaded_methods,
       type_defs.numberconstructors as type_def__number_of_constructors,
       type_defs.numberfields as type_def__number_of_fields,
       type_defs.numbernestedtypes as type_def__number_of_nested_types,
       type_defs.numberinnertypes as type_def__number_of_inner_types,
       type_defs.namingconvention as type_def__naming_convention,
       type_defs.isabstract as type_def__is_abstract,
       type_defs.numberstaticnestedtypes as type_def__number_of_static_nested_types,
       type_defs.isinnerclass as type_def__is_inner,
       type_defs.isstrictfp as type_def__is_strictfp,
       type_defs.staticfieldpercentage as type_def__percentage_of_static_fields,
       type_defs.staticmethodpercentage as type_def__percentage_of_static_methods,
       type_defs.numberstaticblocks as type_def__number_of_static_blocks,
       type_defs.isnestedclass as type_def__is_nested,
       type_defs.isstatic as type_def__is_static,

       method_defs.nodetype_id as method_def__node_id,
       method_defs.methodvisibility as method_def__visibility,
       method_defs.defaultimplementation as method_def__is_default_impl,
       method_defs.isfinal as method_def__is_final,
       method_defs.hasoverride as method_def__is_override,
       method_defs.isstatic as method_def__is_static,
       method_defs.numberparameters as method_def__number_of_parameters,
       method_defs.numbergenerictypes as method_def__number_of_generics,
       method_defs.numberthrows as method_def__number_of_throws,
       method_defs.returntype as method_def__return_type,
       method_defs.numberannotations as method_def__number_of_annotations,
       method_defs.numberstmts as method_def__number_of_statements,
       method_defs.methodlocalvars as method_def__number_of_local_vars,
       method_defs.locvarnaming as method_def__naming_convention_of_local_vars,
       method_defs.namingconvention as method_def__naming_convention,
       method_defs.isabstract as method_def__is_abstract,
       method_defs.numberinnerclasses as method_def__number_of_inner_classes,
       method_defs.isconstructor as method_def__is_constructor,
       method_defs.isstrictfp as method_def__is_strictfp,
       method_defs.isnative as method_def__is_native,
       method_defs.issynchronized as method_def__is_synchronized,
       method_defs.numberofoverloaded as method_def__number_of_times_overloaded,
       method_defs.firstparametertype as method_def__type_of_first_parameter,
       method_defs.secondparametertype as method_def__type_of_second_parameter,
       method_defs.thirdparametertype as method_def__type_of_third_parameter,
       method_defs.user_id as method_def__user_id,
       method_defs.user_class as method_def__user_class,

       param1_type.nodetype_id as method_def__param1_type_node_id,
       param1_type.type_category as method_def__param1_type_syntactic_category,
       param1_type.primitive as method_def__param1_type_is_primitive,
       param1_type.height as method_def__param1_type_height,
       param1_type.parentnode as method_def__param1_type_parent_node_syntactic_category,
       param1_type.parentchild as method_def__param1_type_role_played_in_parent_node,
       param1_type.generics as method_def__param1_type_number_of_generics,
       param1_type.dimensions as method_def__param1_type_number_of_dimensions,

       param2_type.nodetype_id as method_def__param2_type_node_id,
       param2_type.type_category as method_def__param2_type_syntactic_category,
       param2_type.primitive as method_def__param2_type_is_primitive,
       param2_type.height as method_def__param2_type_height,
       param2_type.parentnode as method_def__param2_type_parent_node_syntactic_category,
       param2_type.parentchild as method_def__param2_type_role_played_in_parent_node,
       param2_type.generics as method_def__param2_type_number_of_generics,
       param2_type.dimensions as method_def__param2_type_number_of_dimensions,

       param3_type.nodetype_id as method_def__param3_type_node_id,
       param3_type.type_category as method_def__param3_type_syntactic_category,
       param3_type.primitive as method_def__param3_type_is_primitive,
       param3_type.height as method_def__param3_type_height,
       param3_type.parentnode as method_def__param3_type_parent_node_syntactic_category,
       param3_type.parentchild as method_def__param3_type_role_played_in_parent_node,
       param3_type.generics as method_def__param3_type_number_of_generics,
       param3_type.dimensions as method_def__param3_type_number_of_dimensions,

       return_type.nodetype_id as method_def__return_type_node_id,
       return_type.type_category as method_def__return_type_syntactic_category,
       return_type.primitive as method_def__return_type_is_primitive,
       return_type.height as method_def__return_type_height,
       return_type.parentnode as method_def__return_type_parent_node_syntactic_category,
       return_type.parentchild as method_def__return_type_role_played_in_parent_node,
       return_type.generics as method_def__return_type_number_of_generics,
       return_type.dimensions as method_def__return_type_number_of_dimensions,

       statements.nodetype_id as statements__node_id,
       statements.statement_category as statements__syntactic_category,
       statements.firstchild as statements__syntactic_category_first_child,
       statements.secondchild as statements__syntactic_category_second_child,
       statements.thirdchild as statements__syntactic_category_third_child,
       statements.parentnode as statements__syntactic_category_parent_node,
       statements.height as statements__height,
       statements.depth as statements__depth,
       statements.user_id as statements__user_id,
       statements.user_class as statements__user_class,

       statement_child_1.nodetype_id as expressions__first_child_node_id,
       statement_child_1.expression_category as expressions__first_child_syntactic_category,
       statement_child_1.firstchild as expressions__first_child_syntactic_category_first_child,
       statement_child_1.secondchild as expressions__first_child_syntactic_category_second_child,
       statement_child_1.thirdchild as expressions__first_child_syntactic_category_third_child,
       statement_child_1.parentnode as expressions__first_child_syntactic_category_parent_node,
       statement_child_1.parentchild as expressions_first_child_role_played_in_parent_node,
       statement_child_1.height as expressions__first_child_height,
       statement_child_1.depth as expressions__first_child_depth,

       statement_child_2.nodetype_id as expressions__second_child_node_id,
       statement_child_2.expression_category as expressions__second_child_syntactic_category,
       statement_child_2.firstchild as expressions__second_child_syntactic_category_first_child,
       statement_child_2.secondchild as expressions__second_child_syntactic_category_second_child,
       statement_child_2.thirdchild as expressions__second_child_syntactic_category_third_child,
       statement_child_2.parentnode as expressions__second_child_syntactic_category_parent_node,
       statement_child_2.parentchild as expressions_second_child_role_played_in_parent_node,
       statement_child_2.height as expressions__second_child_height,
       statement_child_2.depth as expressions__second_child_depth,

       statement_child_3.nodetype_id as expressions__third_child_node_id,
       statement_child_3.expression_category as expressions__third_child_syntactic_category,
       statement_child_3.firstchild as expressions__third_child_syntactic_category_first_child,
       statement_child_3.secondchild as expressions__third_child_syntactic_category_second_child,
       statement_child_3.thirdchild as expressions__third_child_syntactic_category_third_child,
       statement_child_3.parentnode as expressions__third_child_syntactic_category_parent_node,
       statement_child_3.parentchild as expressions_third_child_role_played_in_parent_node,
       statement_child_3.height as expressions__third_child_height,
       statement_child_3.depth as expressions__third_child_depth

FROM program as programs
RIGHT JOIN file ON programs.program_id = file.program_id
RIGHT JOIN typedefinition as type_defs ON file.file_id = type_defs.file_id
RIGHT JOIN nodetype AS nt1 ON type_defs.nodetype_id = nt1.parentnode_id
RIGHT JOIN methoddefinition as method_defs ON nt1.nodetype_id = method_defs.nodetype_id
RIGHT JOIN nodes_parent_method as parent_method ON method_defs.nodetype_id = parent_method.node_id_of_parent_method
JOIN statement as statements ON parent_method.node_id = statements.nodetype_id
LEFT JOIN "type" as return_type on method_defs.returntype_id = return_type.nodetype_id
LEFT JOIN "type" as param1_type on method_defs.firsttypeparameter_id = param1_type.nodetype_id
LEFT JOIN "type" as param2_type on method_defs.secondtypeparameter_id = param2_type.nodetype_id
LEFT JOIN "type" as param3_type on method_defs.thirdtypeparameter_id = param3_type.nodetype_id
LEFT JOIN expression as statement_child_1 on statements.firstchildid = statement_child_1.nodetype_id
LEFT JOIN expression as statement_child_2 on statements.secondchildid = statement_child_2.nodetype_id
LEFT JOIN expression as statement_child_3 on statements.thirdchildid = statement_child_3.nodetype_id