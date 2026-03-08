
from .base import OutputWriter


class VTKWriter(OutputWriter):

    def write_output(self, **fields) -> None:
        """
        Write output data in VTK format.
        """
        return