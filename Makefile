# Main Makefile

SM64ENV = sm64env
SM64COOPDX = $(SM64ENV)/sm64coopdx

APPLY_PATCH = apply_patch.sh
ROM_FILE = baserom.us.z64


# Define the target that will copy the ROM file and call the Makefile in SM64COOPDX
all: copy-rom
	@echo "Calling Makefile in $(SM64COOPDX)..."
	$(MAKE) -C $(SM64COOPDX) -j

# Define a target to copy the ROM file into the SM64COOPDX
copy-rom:
	@echo "Copying $(ROM_FILE) to $(SM64COOPDX)..."
	@if [ ! -f $(ROM_FILE) ]; then \
		echo -e "\033[0;31mA Mario 64 ROM is required with the name $(ROM_FILE)\033[0m"; \
		exit 1; \
	fi
# Need it in SM64ENV when the game runs, and also need it in SM64COOPDX to build
	cp $(ROM_FILE) $(SM64COOPDX)/$(ROM_FILE)
	cp $(ROM_FILE) $(SM64ENV)/$(ROM_FILE)

# create-patch:
# 	@echo "Creating patch..."
# 	@cd $(SM64COOPDX) && yes | ./tools/create_patch.sh ../patch.diff

# apply-patch:
# 	@echo "Applying patch..."
# 	-@cd $(SM64COOPDX) && yes | ./tools/apply_patch.sh ../patch.diff

clean:
	@echo "Cleaning up..."
	$(MAKE) -C $(SM64COOPDX) clean

.PHONY: all copy-rom clean
