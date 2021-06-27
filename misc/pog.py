from xml.etree import ElementTree as et
import pandas as pd

class EDIMapper:
    def __init__(self, xml_path, mapper:dict, iterative_keys=None):
        """

        :param xml_path: the path to the XML sheet that we will fill values into.
        :param mapper: dictionary that maps gathered SQL fields to XML files.
                        Has the format {XML field: SQL field}. For nested values,
                        use a nested dictionary.
        :param iterative_keys:
        """
        assert type(mapper) == dict, f"The EDI mapper needs to be of type 'dictionary'. You passed {mapper} " \
                                     f"which is of type {type(mapper)}"
        self.mapper = mapper
        self.iterative_keys = iterative_keys if iterative_keys is not None else set()
        self.xml_template = et.parse(xml_path)
        self.root = self.xml_template.getroot()

    def map_to_xml(self, df:pd.DataFrame):
        for k, v in self.mapper.items():
            df_shape = df.shape[0]
            num_iters = 1
            if k in iterative_keys:
                num_iters = df_shape
                self.xml_template = self.__clear_node(k)
                self.xml_template = self.__create_nodes(k, v, num_iters=num_iters)

            for i in range(num_iters):
                self.xml_template = self.__insert_value(k, v, df, i)



        return self.xml_template

    def __clear_node(self, k):
        template = self.xml_template
        nodes = template.findall(k)
        for node in nodes:
            for n in node:
                node.remove(n)
        return template

    def __create_nodes(self, k, v, num_iters:int):
        template = self.xml_template

        root_nodes = template.findall(k)
        for root_node in root_nodes:
            for i in range(num_iters):
                if type(v) == dict:
                    for nested_k, nested_v in v.items():
                        subelement = et.SubElement(root_node, nested_k)
                        subelement.set('sql_query_row', str(i))
                        template = self.__expand_nodes(subelement, nested_v, template)
                else:
                    subelement = et.SubElement(root_node, str(v))
                    subelement.set('sql_query_row', str(i))
        return template

    def __expand_nodes(self, node, v, template):
        if type(v) != dict:
            et.SubElement(node, str(v))
            return template

        for nested_k, nested_v in v.items():
            subelement = et.SubElement(node, nested_k)
            self.__expand_nodes(subelement, nested_v, template)

        return template

    def __insert_value(self, k, v, df, i=0) -> et:
        """
        private recursive function to add text fields to XML document using the mapper
        :param k:
        :param v:
        :param df:
        :return:
        """

        template = self.xml_template

        if type(v) == dict:
            for nested_k, nested_v in v.items():
                nested_k = k + '/' + nested_k
                template = self.__insert_value(nested_k, nested_v, df, i=i)
            return template

        try:
            template.findall(k)[i].text = str(df[v][i])
        except AttributeError:
            print(f'\tWARNING: Cannot find attribute {k}')
        except KeyError:
            print(f'\tWARNING: KeyError: {k}')

        return template

    def get_string_template(self) -> str:
        """

        :return: string form of .XML file
        """
        template = self.xml_template.getroot()
        template = et.tostring(template, encoding='unicode')
        return template

    def export_xml_to_file(self, outpath:str) -> None:
        """

        :param outpath: path to write the XML file to.
        :return: None
        """
        if not outpath.endswith('.xml'): outpath += '.xml'
        with open(outpath, 'w+') as f:
            f.write(self.get_string_template())

df = pd.DataFrame({
        'order_row_number': [1,2,3],
        'product_number': ['A4', 'B7', 'QQ9'],
        'quantity': [4, 8, 12],
        'int_unit_price': [.69, 1, 10],
        'dropship': ['no', 'yes', 'yes']
    })

test_mapper = {
    'LineItems': {
        'LineItem': {
            'OrderLine': {
                'OrderQty': 'quantity',
                'PurchasePrice': 'int_unit_price',
                'VendorPartNumber': 'product_number',
                'ButtButt': 'order_row_number'
            }
        }
    },
    'Meta': {
        'IsDropShip': 'dropship'
    }
}
iterative_keys = {'LineItems'}
xml_path = '../../ePartsServices/EDI/jci_example.xml'

edi_mapper = EDIMapper(xml_path, mapper=test_mapper, iterative_keys=iterative_keys)
edi_mapper.map_to_xml(df)
with open('../../ePartsServices/EDI/test.xml', 'w+') as f:
    f.write(edi_mapper.get_string_template())