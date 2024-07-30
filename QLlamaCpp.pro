QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++2b

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    common/build-info.cpp \
    common/common.cpp \
    common/console.cpp \
    common/grammar-parser.cpp \
    common/json-schema-to-grammar.cpp \
    common/ngram-cache.cpp \
    common/sampling.cpp \
    common/train.cpp \
    llava/clip.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    QLlamaInference.hpp \
    common/base64.hpp \
    common/common.h \
    common/console.h \
    common/grammar-parser.h \
    common/json-schema-to-grammar.h \
    common/json.hpp \
    common/log.h \
    common/ngram-cache.h \
    common/sampling.h \
    common/stb_image.h \
    common/train.h \
    llava/clip.h \
    llava/llava.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

unix: LIBS += -L/usr/local/lib -lllama -lllava_shared
#win32: LIBS += -LC:\Program Files\Llama\lib -llama -lllava_shared


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    common/CMakeLists.txt \
    common/build-info.cpp.in \
    common/cmake/build-info-gen-cpp.cmake
