Qcore library
# Changelog
(Based on https://wiki.canterbury.ac.nz/download/attachments/58458136/CodeVersioning_v18p2.pdf?version=1&modificationDate=1519269238437&api=v2 )

## [19.5.3] - 2019-09-13 -- Added fault selection file laoder
### Added
    - Function in formats to load fault selection files

## [19.5.2] - 2019-08-22 -- Added cross track distance
### Added
    - Added formula for cross track distance for R_x calculations

## [19.5.1] - 2019-05-13 -- Initial Version
### Changed
    - get_bounds now returns the location of point source earthquakes
    - Validate VM can now validate that srfs for point source realisations are within the VM
    - If incident corners of a VM have the same longitude then the points between them are now correctly interpolated
