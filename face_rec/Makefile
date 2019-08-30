include $(TOPDIR)/rules.mk
include $(INCLUDE_DIR)/kernel.mk

PKG_NAME:=face_rec
PKG_RELEASE:=1

PKG_BUILD_DIR:=$(BUILD_DIR)/$(PKG_NAME)
PKG_INSTALL_DIR:=$(PKG_BUILD_DIR)/ipkg-install

include $(INCLUDE_DIR)/package.mk
include $(INCLUDE_DIR)/cmake.mk

TARGET_CFLAGS += -I$(STAGING_DIR)/usr/include -Wno-error -march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4
TARGET_CPPFLAGS += -I$(STAGING_DIR)/usr/include -Wno-error -march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4
#TARGET_LDFLAGS+= $(STAGING_DIR)/usr/lib/libncnn.a

define Package/face_rec
  SECTION:=Vendor
  CATEGORY:=Vendor
  DEPENDS:=+ncnn +opencv
  TITLE:=Face recognization
endef

define Package/face_rec/description
	Face recognization base on ncnn and mtncnn
endef

define Build/Prepare
	mkdir -p $(PKG_BUILD_DIR)
	$(CP) ./src/* $(PKG_BUILD_DIR)/
endef

define Package/face_rec/install
	$(INSTALL_DIR) $(1)/usr/sbin
	$(INSTALL_BIN) $(PKG_BUILD_DIR)/facedect $(1)/usr/sbin/facedect
	$(INSTALL_BIN) $(PKG_BUILD_DIR)/genfeature $(1)/usr/sbin/genfeature
	$(CP) ./files/* $(1)/
endef

$(eval $(call BuildPackage,face_rec))

