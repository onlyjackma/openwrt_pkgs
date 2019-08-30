include $(TOPDIR)/rules.mk
include $(INCLUDE_DIR)/kernel.mk

PKG_NAME:=ncnn
PKG_RELEASE:=1

PKG_BUILD_DIR:=$(BUILD_DIR)/$(PKG_NAME)
PKG_INSTALL_DIR:=$(PKG_BUILD_DIR)/ipkg-install

include $(INCLUDE_DIR)/package.mk
include $(INCLUDE_DIR)/cmake.mk

TARGET_CFLAGS += -I$(STAGING_DIR)/usr/include -Wno-error
TARGET_CPPFLAGS += -std=c++11 -march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 -fopenmp -D__ARM_NEON
#TARGET_LDFLAGS+= $(STAGING_DIR)/usr/lib/libjson-c.so.2 $(STAGING_DIR)/usr/lib/libcares.so.2
#-DNCNN_STRING=ON
CMAKE_OPTIONS += -DNCNN_INSTALL_SDK=ON -DNCNN_PIXEL=ON -DNCNN_STRING=ON -DNCNN_STDIO=ON
#DEPENDS:=

define Package/ncnn
  SECTION:=libs
  CATEGORY:=Libraries
  TITLE:=NCNN 
endef

define Package/ncnn/description
	NCNN is a high-performance neural network inference framework
endef


# SET ( CMAKE_CXX_FLAGS "-std=c++11 -march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 -fopenmp ${CMAKE_CXX_FLAGS}" )

define Build/Prepare
	mkdir -p $(PKG_BUILD_DIR)
	$(CP) ./src/* $(PKG_BUILD_DIR)/
endef


define Build/InstallDev
	mkdir -p $(1)/usr/include
	$(CP)	$(PKG_INSTALL_DIR)/usr/include/*.h \
		$(1)/usr/include/
	mkdir -p $(1)/usr/lib
	$(CP)	$(PKG_INSTALL_DIR)/usr/lib/libncnn.a \
		$(1)/usr/lib/
endef

define Package/ncnn/install
	$(INSTALL_DIR) $(PKG_INSTALL_DIR)
endef

$(eval $(call BuildPackage,ncnn))

