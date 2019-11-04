import sys, re

chapter_re = re.compile("\s*CHAPTER\s+([0-9]+|[CLXVI]+)", flags=re.IGNORECASE)

def isChapter(line):
    return chapter_re.search(line)

# Return true if more than half the words on this line are uppercase
# Don't count single capital letters (e.g. the word I, initials) as uppercase
# TODO: initials followed by periods still filtered out
def isUppercase(line):
    ws = [w.strip() for w in line.split()]
    uppercase = 0
    for w in ws:
        if w.upper() == w and len(w) > 1:
            uppercase += 1

    if uppercase >= len(ws) * 0.5:
        return True

    return False

def isEmpty(line):
    if line.strip() == "":
        return True
    return False


# Return false if the provided line is some sort of meta-line that is not part
# of the book.
# e.g. chapter headings, all-caps sentences, attributions.
def isMetaline(line):
    if isEmpty(line):
        return True

    if isChapter(line):
        print("chap", line)
        return True

    if isUppercase(line):
        print("^^", line)
        return True

    return False

def main():
    filenames = sys.argv[1:]

    start_re = re.compile("\*\*\*\s?START OF (THIS|THE) PROJECT GUTENBERG EBOOK")
    end_re = re.compile("\*\*\*\s?END OF (THIS|THE) PROJECT GUTENBERG EBOOK")

    # The final *** might be on another line, so we need to separately check for
    # the ending token
    header_end_re = re.compile("\*\*\*\s*$")


    for filename in filenames:

        pcs = filename.split("/")
        pcs[0] = "clean"
        outFilename = "/".join(pcs)
        book = pcs[-1]
        print("\n\n\n================= %s ===============" % book)

        # Open the input and output files
        f = open(filename, "r", encoding="utf-8")
        out = open(outFilename, "w", encoding="utf-8")

        #
        seenStart = False
        seeingStart = False
        for i, line in enumerate(f):

            # Check if the EBOOK START header is beginning
            if start_re.search(line):
                seenStart = True
                # Check for ending token
                if not header_end_re.search(line.strip()):
                    seeingStart = True
                print(line)
                continue

            # If the EBOOK START header already began but has not yet ended
            if seeingStart:
                print("seeing start", line)
                # If this is the end of the start header, finish the header, move to next line
                if header_end_re.search(line.strip()):
                    seeingStart = False
                continue

            # Check for the EBOOK END header
            if end_re.search(line):
                print("end:", line)
                break

            # Ignore all text before the EBOOK START header
            if not seenStart:
                continue

            # Normal text:
            if isMetaline(line):
                continue


            out.write(line)

        # Close the input and output files
        f.close()
        out.close()


if __name__ == '__main__':
    main()
