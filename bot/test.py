import re

uncleaned_message = "<@1177648262971412551> !roll"
message = re.sub(r'<@!?\d+>', '', uncleaned_message)
print(message)