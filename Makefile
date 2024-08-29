# Main Makefile

# Define the subfolder path
SUBFOLDER = env/sm64coopdx
ROM_FILE = baserom.us.z64

# Define the target that will copy the ROM file and call the Makefile in the subfolder
all: copy-rom
	@echo "Calling Makefile in $(SUBFOLDER)..."
	$(MAKE) -C $(SUBFOLDER) -j

# Define a target to copy the ROM file into the subfolder
copy-rom:
	@echo "Copying $(ROM_FILE) to $(SUBFOLDER)..."
	cp $(ROM_FILE) $(SUBFOLDER)/$(ROM_FILE)

clean:
	@echo "Cleaning up..."
	$(MAKE) -C $(SUBFOLDER) clean

.PHONY: all copy-rom clean
