import json
import os
from math import sqrt
import numpy as np

from collections import MutableSequence

class AnnoList(MutableSequence):
    """Define a list format, which I can customize"""
    TYPE_INT32 = 5;
    TYPE_FLOAT = 2;
    TYPE_STRING = 9;

    def __init__(self, data=None):
        super(AnnoList, self).__init__()

        self.attribute_desc = {};
        self.attribute_val_to_str = {};

        if not (data is None):
            self._list = list(data)
        else:
            self._list = list()

    def add_attribute(self, name, dtype):
        _adesc = AnnoList_pb2.AttributeDesc();
        _adesc.name = name;
        if self.attribute_desc:
            _adesc.id = max((self.attribute_desc[d].id for d in self.attribute_desc)) + 1;
        else:
            _adesc.id = 0;

        if dtype == int:
            _adesc.dtype = AnnoList.TYPE_INT32;
        elif dtype == float or dtype == np.float32:
            _adesc.dtype = AnnoList.TYPE_FLOAT;
        elif dtype == str:
            _adesc.dtype = AnnoList.TYPE_STRING;
        else:
            print "unknown attribute type: ", dtype
            assert(False);

        #print "adding attribute: {}, id: {}, type: {}".format(_adesc.name, _adesc.id, _adesc.dtype);
        self.attribute_desc[name] = _adesc;

    def add_attribute_val(self, aname, vname, val):
        # add attribute before adding string corresponding to integer value
        assert(aname in self.attribute_desc);

        # check and add if new
        if all((val_desc.id != val for val_desc in self.attribute_desc[aname].val_to_str)):
            val_desc = self.attribute_desc[aname].val_to_str.add()
            val_desc.id = val;
            val_desc.s = vname;

        # also add to map for quick access
        if not aname in self.attribute_val_to_str:
            self.attribute_val_to_str[aname] = {};

        assert(not val in self.attribute_val_to_str[aname]);
        self.attribute_val_to_str[aname][val] = vname;


    def attribute_get_value_str(self, aname, val):
        if aname in self.attribute_val_to_str and val in self.attribute_val_to_str[aname]:
            return self.attribute_val_to_str[aname][val];
        else:
            return str(val);

    def save(self, fname):
        save(fname, self);

    #MA: list interface
    def __len__(self):
        return len(self._list)

    def __getitem__(self, ii):
        if isinstance(ii, slice):
            res = AnnoList();
            res.attribute_desc = self.attribute_desc;
            res._list = self._list[ii]
            return res;
        else:
            return self._list[ii]

    def __delitem__(self, ii):
        del self._list[ii]

    def __setitem__(self, ii, val):
        return self._list[ii]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return """<AnnoList %s>""" % self._list

    def insert(self, ii, val):
        self._list.insert(ii, val)

    def append(self, val):
        list_idx = len(self._list)
        self.insert(list_idx, val)

class AnnoRect(object):
    def __init__(self, x1=-1, y1=-1, x2=-1, y2=-1):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.score = -1.0
        self.scale = -1.0
        self.articulations =[]
        self.viewpoints =[]
        self.d3 = []

        self.silhouetteID = -1
        self.classID = -1
        self.track_id = -1

        self.point = [];
        self.at = {};

    def width(self):
        return abs(self.x2-self.x1)

    def height(self):
        return abs(self.y2-self.y1)

    def centerX(self):
        return (self.x1+self.x2)/2.0

    def centerY(self):
        return (self.y1+self.y2)/2.0

    def left(self):
        return min(self.x1, self.x2)

    def right(self):
        return max(self.x1, self.x2)

    def top(self):
        return min(self.y1, self.y2)

    def bottom(self):
        return max(self.y1, self.y2)

    def forceAspectRatio(self, ratio, KeepHeight = False, KeepWidth = False):
        """force the Aspect ratio"""
        if KeepWidth or ((not KeepHeight) and self.width() * 1.0 / self.height() > ratio):
            # extend height
            newHeight = self.width() * 1.0 / ratio
            self.y1 = (self.centerY() - newHeight / 2.0)
            self.y2 = (self.y1 + newHeight)
        else:
            # extend width
            newWidth = self.height() * ratio
            self.x1 = (self.centerX() - newWidth / 2.0)
            self.x2 = (self.x1 + newWidth)

    def clipToImage(self, min_x, max_x, min_y, max_y):
        self.x1 = max(min_x, self.x1)
        self.x2 = max(min_x, self.x2)
        self.y1 = max(min_y, self.y1)
        self.y2 = max(min_y, self.y2)
        self.x1 = min(max_x, self.x1)
        self.x2 = min(max_x, self.x2)
        self.y1 = min(max_y, self.y1)
        self.y2 = min(max_y, self.y2)

    def printContent(self):
        print "Coords: ", self.x1, self.y1, self.x2, self.y2
        print "Score: ", self.score
        print "Articulations: ", self.articulations
        print "Viewpoints: ", self.viewpoints
        print "Silhouette: ", self.silhouetteID

    def ascii(self):
        r = "("+str(self.x1)+", "+str(self.y1)+", "+str(self.x2)+", "+str(self.y2)+")"
        if (self.score!=-1):
            r = r + ":"+str(self.score)
        if (self.silhouetteID !=-1):
            ri = r + "/"+str(self.silhouetteID)
        return r

    def writeIDL(self, file):
        file.write(" ("+str(self.x1)+", "+str(self.y1)+", "+str(self.x2)+", "+str(self.y2)+")")
        if (self.score!=-1):
            file.write(":"+str(self.score))
        if (self.silhouetteID !=-1):
            file.write("/"+str(self.silhouetteID))

    def writeJSON(self):
        jdoc = {"x1": self.x1, "x2": self.x2, "y1": self.y1, "y2": self.y2}

        if (self.score != -1):
            jdoc["score"] = self.score
        return jdoc

    def sortCoords(self):
        if (self.x1>self.x2):
            self.x1, self.x2 = self.x2, self.x1
        if (self.y1>self.y2):
            self.y1, self.y2 = self.y2, self.y1

    def rescale(self, factor):
        self.x1=(self.x1*float(factor))
        self.y1=(self.y1*float(factor))
        self.x2=(self.x2*float(factor))
        self.y2=(self.y2*float(factor))

    def resize(self, factor, factor_y = None):
        w = self.width()
        h = self.height()
        if factor_y is None:
            factor_y = factor
        centerX = float(self.x1+self.x2)/2.0
        centerY = float(self.y1+self.y2)/2.0
        self.x1 = (centerX - (w/2.0)*factor)
        self.y1 = (centerY - (h/2.0)*factor_y)
        self.x2 = (centerX + (w/2.0)*factor)
        self.y2 = (centerY + (h/2.0)*factor_y)


    def intersection(self, other):
        self.sortCoords()
        other.sortCoords()

        if(self.x1 >= other.x2):
            return (0, 0)
        if(self.x2 <= other.x1):
            return (0, 0)
        if(self.y1 >= other.y2):
            return (0, 0)
        if(self.y2 <= other.y1):
            return (0, 0)

        l = max(self.x1, other.x1);
        t = max(self.y1, other.y1);
        r = min(self.x2, other.x2);
        b = min(self.y2, other.y2);
        return (r - l, b - t)

        #Alternate implementation
        #nWidth  = self.x2 - self.x1
        #nHeight = self.y2 - self.y1
        #iWidth  = max(0,min(max(0,other.x2-self.x1),nWidth )-max(0,other.x1-self.x1))
        #iHeight = max(0,min(max(0,other.y2-self.y1),nHeight)-max(0,other.y1-self.y1))
        #return (iWidth, iHeight)

    def cover(self, other):
        nWidth = self.width()
        nHeight = self.height()
        iWidth, iHeight = self.intersection(other)
        return float(iWidth * iHeight) / float(nWidth * nHeight)

    def overlap_pascal(self, other):
        self.sortCoords()
        other.sortCoords()

        nWidth  = self.x2 - self.x1
        nHeight = self.y2 - self.y1
        iWidth, iHeight = self.intersection(other)
        interSection = iWidth * iHeight

        union = self.width() * self.height() + other.width() * other.height() - interSection

        overlap = interSection * 1.0 / union
        return overlap

    def isMatchingPascal(self, other, minOverlap):
        overlap = self.overlap_pascal(other)
        if (overlap >= minOverlap and (self.classID == -1 or other.classID == -1 or self.classID == other.classID)):
            return 1
        else:
            return 0

    def distance(self, other, aspectRatio=-1, fixWH='fixheight'):
        if (aspectRatio!=-1):
            if (fixWH=='fixwidth'):
                dWidth  = float(self.x2 - self.x1)
                dHeight = dWidth / aspectRatio
            elif (fixWH=='fixheight'):
                dHeight = float(self.y2 - self.y1)
                dWidth  = dHeight * aspectRatio
        else:
            dWidth  = float(self.x2 - self.x1)
            dHeight = float(self.y2 - self.y1)

        xdist   = (self.x1 + self.x2 - other.x1 - other.x2) / dWidth
        ydist   = (self.y1 + self.y2 - other.y1 - other.y2) / dHeight

        return sqrt(xdist*xdist + ydist*ydist)

    def isMatchingStd(self, other, coverThresh, overlapThresh, distThresh, aspectRatio=-1, fixWH=-1):
        cover = other.cover(self)
        overlap = self.cover(other)
        dist = self.distance(other, aspectRatio, fixWH)

        #if(self.width() == 24 ):
        #print cover, " ", overlap, " ", dist
        #print coverThresh, overlapThresh, distThresh
        #print (cover>=coverThresh and overlap>=overlapThresh and dist<=distThresh)

        if (cover>=coverThresh and overlap>=overlapThresh and dist<=distThresh and self.classID == other.classID):
            return 1
        else:
            return 0

    def isMatching(self, other, style, coverThresh, overlapThresh, distThresh, minOverlap, aspectRatio=-1, fixWH=-1):
        #choose matching style
        if (style == 0):
            return self.isMatchingStd(other, coverThresh, overlapThresh, distThresh, aspectRatio=-1, fixWH=-1)

        if (style == 1):
            return self.isMatchingPascal(other, minOverlap)

    def addToXML(self, node, doc): # no Silhouette yet
        rect_el = doc.createElement("annorect")
        for item in "x1 y1 x2 y2 score scale track_id".split():
            coord_el = doc.createElement(item)
            coord_val = doc.createTextNode(str(self.__getattribute__(item)))
            coord_el.appendChild(coord_val)
            rect_el.appendChild(coord_el)

        articulation_el = doc.createElement("articulation")
        for articulation in self.articulations:
            id_el = doc.createElement("id")
            id_val = doc.createTextNode(str(articulation))
            id_el.appendChild(id_val)
            articulation_el.appendChild(id_el)
        if(len(self.articulations) > 0):
            rect_el.appendChild(articulation_el)

        viewpoint_el    = doc.createElement("viewpoint")
        for viewpoint in self.viewpoints:
            id_el = doc.createElement("id")
            id_val = doc.createTextNode(str(viewpoint))
            id_el.appendChild(id_val)
            viewpoint_el.appendChild(id_el)
        if(len(self.viewpoints) > 0):
            rect_el.appendChild(viewpoint_el)

        d3_el    = doc.createElement("D3")
        for d in self.d3:
            id_el = doc.createElement("id")
            id_val = doc.createTextNode(str(d))
            id_el.appendChild(id_val)
            d3_el.appendChild(id_el)
        if(len(self.d3) > 0):
            rect_el.appendChild(d3_el)

        if self.silhouetteID != -1:
            silhouette_el    = doc.createElement("silhouette")
            id_el = doc.createElement("id")
            id_val = doc.createTextNode(str(self.silhouetteID))
            id_el.appendChild(id_val)
            silhouette_el.appendChild(id_el)
            rect_el.appendChild(silhouette_el)

        if self.classID != -1:
            class_el    = doc.createElement("classID")
            class_val = doc.createTextNode(str(self.classID))
            class_el.appendChild(class_val)
            rect_el.appendChild(class_el)

        if len(self.point) > 0:
            annopoints_el = doc.createElement("annopoints")

            for p in self.point:
                point_el = doc.createElement("point");

                point_id_el = doc.createElement("id");
                point_id_val = doc.createTextNode(str(p.id));
                point_id_el.appendChild(point_id_val);
                point_el.appendChild(point_id_el);

                point_x_el = doc.createElement("x");
                point_x_val = doc.createTextNode(str(p.x));
                point_x_el.appendChild(point_x_val);
                point_el.appendChild(point_x_el);

                point_y_el = doc.createElement("y");
                point_y_val = doc.createTextNode(str(p.y));
                point_y_el.appendChild(point_y_val);
                point_el.appendChild(point_y_el);

                annopoints_el.appendChild(point_el);

            rect_el.appendChild(annopoints_el);

        node.appendChild(rect_el)

class Annotation(object):

    def __init__(self):
        self.imageName = ""
        self.imagePath = ""
        self.rects =[]
        self.frameNr = -1

    def clone_empty(self):
        new = Annotation()
        new.imageName = self.imageName
        new.imagePath = self.imagePath
        new.frameNr   = self.frameNr
        new.rects     = []
        return new

    def filename(self):
        return os.path.join(self.imagePath, self.imageName)

    def printContent(self):
        print "Name: ", self.imageName
        for rect in self.rects:
            rect.printContent()

    def writeIDL(self, file):
        if (self.frameNr == -1):
            file.write("\""+os.path.join(self.imagePath, self.imageName)+"\"")
        else:
            file.write("\""+os.path.join(self.imagePath, self.imageName)+"@%d\"" % self.frameNr)

        if (len(self.rects)>0):
            file.write(":")
        i=0
        for rect in self.rects:
            rect.writeIDL(file)
            if (i+1<len(self.rects)):
                file.write(",")
            i+=1

    def writeJSON(self):
        jdoc = {}
        jdoc['image_path'] = os.path.join(self.imagePath, self.imageName)
        jdoc['rects'] = []
        for rect in self.rects:
            jdoc['rects'].append(rect.writeJSON())
        return jdoc

    def addToXML(self, node, doc): # no frame# yet
        annotation_el = doc.createElement("annotation")
        img_el = doc.createElement("image")
        name_el = doc.createElement("name")
        name_val = doc.createTextNode(os.path.join(self.imagePath, self.imageName))
        name_el.appendChild(name_val)
        img_el.appendChild(name_el)

        if(self.frameNr != -1):
            frame_el = doc.createElement("frameNr")
            frame_val = doc.createTextNode(str(self.frameNr))
            frame_el.appendChild(frame_val)
            img_el.appendChild(frame_el)

        annotation_el.appendChild(img_el)
        for rect in self.rects:
            rect.addToXML(annotation_el, doc)
        node.appendChild(annotation_el)


    def sortByScore(self, dir="ascending"):
        if (dir=="descending"):
            self.rects.sort(cmpAnnoRectsByScoreDescending)
        else:
            self.rects.sort(cmpAnnoRectsByScore)

    def __getitem__(self, index):
        return self.rects[index]
def parseJSON(filename):
    filename = os.path.realpath(filename)
    name, ext = os.path.splitext(filename)
    assert ext == '.json'

    annotations = AnnoList([])
    with open(filename, 'r') as f:
        jdoc = json.load(f)

    for annotation in jdoc:
        anno = Annotation()
        anno.imageName = annotation["image_path"]

        rects = []
        for annoRect in annotation["rects"]:
            rect = AnnoRect()

            rect.x1 = annoRect["x1"]
            rect.x2 = annoRect["x2"]
            rect.y1 = annoRect["y1"]
            rect.y2 = annoRect["y2"]
            if "score" in annoRect:
                rect.score = annoRect["score"]

            rects.append(rect)

        anno.rects = rects
        annotations.append(anno)

    return annotations

def parse(filename, abs_path=False):
    #print "Parsing: ", filename
    name, ext = os.path.splitext(filename)

    if(ext == ".json"):
        annolist = parseJSON(filename)
    else:
        annolist = AnnoList([]);

    if abs_path:
        basedir = os.path.dirname(os.path.abspath(filename))
        for a in annolist:
            a.imageName = basedir + "/" + os.path.basename(a.imageName)

    return annolist
