from src.data_normalizer.DataNormalizer import DataNormalizer


class IdentityNormalizer(DataNormalizer):

    def normalize(self, data):
        return data

    def un_normalize(self, data):
        return data
