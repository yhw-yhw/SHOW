from inspect import Parameter, signature


class ObjectFactory:
    def add_to_parser(self, parser):
        raise NotImplementedError()

    def from_dict(self, arguments):
        raise NotImplementedError()


class EmptyFactory(ObjectFactory):
    """An EmptyFactory simply returns the passed in object."""
    def __init__(self, _object):
        self._object = _object

    def add_to_parser(self, parser):
        pass

    def from_dict(self, arguments):
        return self._object


class FactoryList(ObjectFactory):
    def __init__(self, factories):
        self.factories = factories

    def add_to_parser(self, parser):
        for f in self.factories:
            f.add_to_parser(parser)

    def from_dict(self, arguments):
        return [
            f.from_dict(arguments)
            for f in self.factories
        ]


class CallableFactory(ObjectFactory):
    """CallableFactory creates an ObjectFactory instance from a callable using
    Python's reflection to define the arguments, argument types and default
    parameters."""
    def __init__(self, func, namespace=""):
        self._func = func
        self._signature = signature(self._func)
        self._namespace = namespace

    @property
    def arg_pattern(self):
        return (
            "{}"
            if self._namespace == ""
            else "{}_{{}}".format(self._namespace)
        )

    def add_to_parser(self, parser):
        arg_pattern = self.arg_pattern
        for parameter in self._signature.parameters.values():
            type = (
                parameter.annotation
                if parameter.annotation is not Parameter.empty
                else str
            )
            default = (
                parameter.default
                if parameter.default is not Parameter.empty
                else None
            )

            parser.add_argument(
                "--" + arg_pattern.format(parameter.name),
                type=type,
                default=default,
                help="{}:{!r} = {}".format(parameter.name, type, default)
            )

    def from_dict(self, arguments):
        # Collect the *args and **kwargs for calling the function
        args = []
        kwargs = {}

        arg_pattern = self.arg_pattern
        for parameter in self._signature.parameters.values():
            # Assemble the key
            key = arg_pattern.format(parameter.name)

            # And fetch the value or the default
            value = parameter.default
            if key in arguments:
                value = arguments[key]
            if value is Parameter.empty:
                raise RuntimeError(("{} parameter is required but a value "
                                    "was not provided").format(key))
            value = (
                parameter.annotation(value)
                if parameter.annotation is not Parameter.empty
                else value
            )

            # Put the argument to the correct place
            if parameter.kind == Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                kwargs[parameter.name] = value

        # Now we can simply call the function with and create whatever we were
        # meant to create
        return self._func(*args, **kwargs)