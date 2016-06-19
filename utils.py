# coding: utf-8

class Utils:    
    @staticmethod
    def SortMapByValue(d):
        return sorted(d.items(), key=lambda x: -x[1])

    @staticmethod
    def TopKeysByValue(map_item_score, topK, ignoreKeys):
        if ignoreKeys is None:
            ignoreSet = set()
        else:
            ignoreSet = set(ignoreKeys)
            
        # Another implementation that first sorting.
        topEntities = Utils.SortMapByValue(map_item_score)
        topKeys = []
        for entry in topEntities:
            if len(topKeys) >= topK:
                break
            if not entry[0] in ignoreSet:
                topKeys.append(entry[0])
        return topKeys
    


