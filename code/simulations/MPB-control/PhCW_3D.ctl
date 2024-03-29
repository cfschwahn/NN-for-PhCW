
;********************************************************************  
;*                                                                  *
;*   MPB ctl file to calculate the bandstructure, group velocity    *  
;*   and mode shape parameters (rho and gamma), for membraned PHC   *  
;*   waveguides. Written by Daryl Beggs and Sebastian Schulz        *
;*                                                                  *
;********************************************************************
(if (= k-split-index 0)  ;only print this info out on the first node of mbi-split calculations
    (begin
      (print "********************* h e l l o ************************** \n")
      (print "PHOTONIC CRYSTAL SLAB DISPERSION ENGINEERED WAVEGUIDE CALCULATION \n")
      (print "VERSION 1 - Daryl Beggs 26 August 2010 \n")
      (print "Script by Daryl Beggs - please see him for help and modifications \n\n")
      (print "Default behavoiur gives a silicon W1 waveguide with air holes \n")
      (print "Parameters that you can alter are listed below \n")
      (print "\ncalculation-type is a code for what you want to calculate (default 1) \n")
      (print "calculation-type=0 gives basic calculation of bandstr \n")
      (print "calculation-type=1 gives bandstr plus group velocities \n")
      (print "calculation-type=2 gives bandstr plus output fields profiles \n")
      (print "calculation-type=3 gives bandstr plus group velocities plus output fields profiles \n\n")
      (print "calculation-type=4 gives bandstr plus group velocities plus calculate integrals \n")
      (print "calculation-type=5 gives bandstr plus group velocities plus calculate integrals plus output fields profiles \n\n")
      (print "index-slab is the index of the slab material (default 3.48) \n")
      (print "index-holes is the index of the holes and cladding (default 1) \n")
      (print "a1nm is the lattice constant in nm (default 400nm) \n")
      (print "a2nm is the lattice constant in nm along the waveguide (default a1nm) \n")
      (print "hslabnm is the height of the slab in nm (default 220nm) \n")
      (print "r0 is the backgraound radius of holes in units of a1 (default 0.3) \n")
      (print "\ndispersion engineering by altering first, second and third row of holes \n")
      (print "r1 is radius of first row of holes (default r0) \n")
      (print "r2 is radius of second row of holes (default r0) \n")
      (print "r3 is radius of third row of holes (default r0) \n")
      (print "s1 is shift of first row of holes inwards (default 0) \n")
      (print "s2 is shift of second row of holes inwards (default 0) \n")
      (print "s3 is shift of third row of holes inwards (default 0) \n")
      (print "p1 is shift of first row of holes along the waveguide (default 0) \n")
      (print "p2 is shift of second row of holes along the waveguide (default 0) \n")
      (print "p3 is shift of third row of holes along the waveguide (default 0) \n")
      (print "w is width of waveguide (i.e. Ww) in units of a*sqrt(3) (i.e. W2 is NOT 2 rows missing) (default 1) \n")
      (print "\nres is the resolution of the grid in-plane (default 16) \n")
      (print "resslab is the resolution of the grid out-of-plane (defualt 30) \n")
      (print "W1band is the band index number of interest (default 23) \n")
      (print "numbands is the number of band to calcultae and must be greater than W1band (default 30) \n")
      (print "Ks is the K-vector to start the bandstructure calculations from (default 0.3) \n")
      (print "Ke is the K-vector to end the bandstructure calculations with (default 0.5) \n")
      (print "Kinterp is the number of K-vector values between Ks and 0.5 (default 19) \n")
      );end begin
    );end if


(print "\n**************** CALCULATION " (+ 1 k-split-index) " of " k-split-num " BEGINS HERE ***************** \n")

(define-param calculation-type 1)

(print "CALCULATION TYPE " calculation-type ": ")
(if (= calculation-type 0)
    (print "basic calculation of bandstr \n"))
(if (= calculation-type 1)
    (print "bandstr plus group velocities \n"))
(if (= calculation-type 2)
    (print "bandstr plus output fields profiles \n"))
(if (= calculation-type 3)
    (print "bandstr plus group velocities plus output fields profiles \n"))
(if (= calculation-type 4)
    (print "bandstr plus group velocities plus calculate integrals \n"))
(if (= calculation-type 5)
    (print "bandstr plus group velocities plus calculate integrals plus output fields profiles \n"))

;*****************************************************************
;define parameters of W1 lattice
(define-param r0 0.30)    ;radius of holes in terms of a1
(define-param a1nm 400.0) ;a1 (nm) is period in lateral direction
(define-param a2nm a1nm)  ;a2 (nm) is period along waveguide
(define-param hslabnm 220.0) ;thickness of slab (nm)
;values above are in nm - convert to relative to a1 below
(define a1 1.0)      ;a1 is period in lateral direction 
(define a2 (/ a2nm a1nm))      ;a2 is period in waveguide direction 
(define hslab (/ hslabnm a1nm))
;define parameters of W1 lattice end
;*****************************************************************

;*****************************************************************
;define parameters for engineering the dispersion using first three rows of holes or waveguide width
;s1,s2,s3 are shifts of holes inwards (towards the waveguide, proportional to a1)
;p1,p2,p3 are shifts of holes alonmg the waveguide (proportional to a2)
;r1,r2,r3 are radii of holes
;all default to give a W1
(define-param w 1)     ;width of waveguide (i.e. Ww) in units of a*sqrt(3) (i.e. W2 is NOT 2 rows missing)
;******************************************************
(define-param r1 r0)   ;r1 is radius of first row of holes
(define-param r2 r0)   ;r2 is radius of second row of holes
(define-param r3 r0)   ;r3 is radius of third row of holes
(define-param s1 0.0)  ;s1 is shift of first row of holes inwards
(define-param s2 0.0)  ;s2 is shift of second row of holes inwards
(define-param s3 0.0)  ;s3 is shift of third row of holes inwards
(define-param p1 0.0)  ;p1 is shift of first row of holes along the waveguide
(define-param p2 0.0)  ;p2 is shift of second row of holes along the waveguide
(define-param p3 0.0)  ;p3 is shift of third row of holes along the waveguide
;define parameters for engineering the dispersion using first three rows of holes or waveguide width end
;*****************************************************************

;*****************************************************************
;print parameters to screen
(print "lattice constant: a1 = " a1nm "nm = " a1 "*a1 \n")
(print "lattice constant along the waveguide: a2 = " a2nm "nm = " a2 "*a1 \n")
(print "height of the slab: hslab = " hslabnm "nm = " hslab "*a1 \n")
(print "width of the waveguide: w = " (* a1nm w (sqrt 3)) "nm = " w "*a1*sqrt(3) \n")
(print "radius of 1st row of holes: r1 = " (* a1nm r1) "nm = " r1 "*a1 \n")
(print "radius of 2nd row of holes: r2 = " (* a1nm r2) "nm = " r2 "*a1 \n")
(print "radius of 3rd row of holes: r3 = " (* a1nm r3) "nm = " r3 "*a1 \n")
(print "shift inwards of 1st row of holes: s1 = " (* a1nm s1) "nm = " s1 "*a1 \n")
(print "shift inwards of 2nd row of holes: s2 = " (* a1nm s2) "nm = " s2 "*a1 \n")
(print "shift inwards of 3rd row of holes: s3 = " (* a1nm s3) "nm = " s3 "*a1 \n")
(print "shift alongwards of 1st row of holes: p1 = " (* a2nm p1) "nm = " p1 "*a2 \n")
(print "shift alongwards of 2nd row of holes: p2 = " (* a2nm p2) "nm = " p2 "*a2 \n")
(print "shift alongwards of 3rd row of holes: p3 = " (* a2nm p3) "nm = " p3 "*a2 \n")
(print "radius of holes r0 = " r0 " a \n")
;print parameters to screen end
;*****************************************************************

;*****************************************************************
;define material and other parameters begin
(define nSiO2 1.44)  ;set refractive index of silicon dioxide
(define SiO2 (make dielectric (epsilon (expt nSiO2 2))))
(define nSi 3.48)    ;set refractive index of silicon
(define Si (make dielectric (epsilon (expt nSi 2))))
(define-param index-slab nSi)   ;default Silicon
(define-param index-holes 1.0)  ;the cladding is also set to index-holes (default 1.0)
(define hole-material (make dielectric (epsilon (expt index-holes 2))))
(define slab-material (make dielectric (epsilon (expt index-slab 2))))
(print "Slab index set to " index-slab "\n")
(print "Hole index set to " index-holes "\n")
(print "Slab material set to " slab-material "\n")
(print "Hole material set to " hole-material "\n")
(set! default-material hole-material)  ; set default material to be hole-material (this defines the cladding)
(print "Default material set to " default-material "\n")
;note - circles are infilled with hole-material
(set! filename-prefix "")
;define material and other parameters end
;*****************************************************************

;*****************************************************************
;define lattice size and geometry begin
(define Zint 4)
(set! geometry-lattice (make lattice (size a2 (* (+ 10 w) (sqrt 3)) (* hslab Zint))
                         (basis1 1 0 0)
                         (basis2 0 1 0)
			 (basis3 0 0 1)))
(define-param res 16)  ;resolution parameter in plane (default to 16)
(define-param resslab 30) ;resolution parameter out of plane (defualt to 30)
(define-param restheta 360) ;resolution for line integrals around a circle (defualt 360)
;define edges of supercell
(define xs (* -0.5 (vector3-x (object-property-value geometry-lattice 'size))))
(define xe (* 0.5 (vector3-x (object-property-value geometry-lattice 'size))))
(define ys (* -0.5 (vector3-y (object-property-value geometry-lattice 'size))))
(define ye (* 0.5 (vector3-y (object-property-value geometry-lattice 'size))))
(define zs (* -0.5 (vector3-z (object-property-value geometry-lattice 'size))))
(define ze (* 0.5 (vector3-z (object-property-value geometry-lattice 'size))))
(print "Supercell: " xs " " xe " " ys " " ye " " zs " " ze "\n")
;rescale resolution to ensure even number of grid points in all directions
(define resx res)
(if (odd? (inexact->exact (ceiling (* res (- xe xs)))))
    (set! resx (+ tolerance (/ (ceiling (* res (- xe xs))) (- xe xs)))) 
    )
(define resy res)
(if (odd? (inexact->exact (ceiling (* res (- ye ys)))))
    (set! resy (+ tolerance (/ (ceiling (* res (- ye ys))) (- ye ys))))
    )
(define resz resslab)
(if (odd? (inexact->exact (ceiling (* resslab (- ze zs)))))
    (set! resz (+ tolerance (/ (ceiling (* resslab (- ze zs))) (- ze zs))))
    )
(define integration-resolution (vector3 1 restheta resz))
(set! resolution (vector3 resx resy resz))
(set! mesh-size 8)
(define-param numbands 30)  ;number of bands defaults to 30
(set! num-bands numbands)
(print "resolution = " resolution "\n")
(print "mesh size = " mesh-size "\n")
(print "number of bands = " num-bands "\n")
(define rhole r0)
(define hhole (* hslab 1.001))   ;make cylinder 0.1% taller than slab
(print "hieght of holes = " hhole "\n")



;define first row with shift s1,p1,r1
(define holes-list-first-row-a (list
				(make cylinder   ;1st row
				  (center (* (+ p1 0.5) a2) (+ (* -0.5 (+ 0 w) (sqrt 3)) s1) 0) 
				  (radius r1) 
				  (height hhole) 
				  (material hole-material))
				))
(define holes-list-first-row-b (list
				(make cylinder  ;first row
				  (center (* (+ p1 0.5) a2) (- (* 0.5 (+ 0 w) (sqrt 3)) s1) 0) 
				  (radius r1) 
				  (height hhole) 
				  (material hole-material))
				))
;define second row with shift s2,p2,r2
(define holes-list-second-row-a (list
				 (make cylinder  ;2nd row
				   (center (* p2 a2) (+ (* -0.5 (+ 1 w) (sqrt 3)) s2) 0) 
				   (radius r2) 
				   (height hhole) 
				   (material hole-material))
				 ))
(define holes-list-second-row-b (list
				 (make cylinder ;second row
				   (center (* p2 a2) (- (* 0.5 (+ 1 w) (sqrt 3)) s2) 0) 
				   (radius r2) 
				   (height hhole) 
				   (material hole-material))
				 ))
;define second row with shift s3,p3,r3
(define holes-list-third-row-a (list
				(make cylinder ;3rd row
				  (center (* (+ p3 0.5) a2) (+ (* -0.5 (+ 2 w) (sqrt 3)) s3) 0) 
				  (radius r3) 
				  (height hhole) 
				  (material hole-material))
				))
(define holes-list-third-row-b (list
				(make cylinder ;third row
				  (center (* (+ p3 0.5) a2) (- (* 0.5 (+ 2 w) (sqrt 3)) s3) 0) 
				  (radius r3) 
				  (height hhole) 
				  (material hole-material))
				))
;define remaining rows
(define holes-list-remaining-rows-a (list
				     (make cylinder
				       (center (* 0.5 a2) (* -0.5 (+ 10 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center 0 (* -0.5 (+ 9 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center (* 0.5 a2) (* -0.5 (+ 8 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center 0 (* -0.5 (+ 7 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center (* 0.5 a2) (* -0.5 (+ 6 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center 0 (* -0.5 (+ 5 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center (* 0.5 a2) (* -0.5 (+ 4 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder    ;4th row
				       (center 0 (* -0.5 (+ 3 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     ))
					;3rd row here
					;2nd row here
					;1st row here
					;centre row here - remove for W1
					;first row here
					;second row here
					;third row here
(define holes-list-remaining-rows-b (list
				     (make cylinder    ;forth row
				       (center 0 (* 0.5 (+ 3 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center (* 0.5 a2) (* 0.5 (+ 4 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center 0 (* 0.5 (+ 5 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center (* 0.5 a2) (* 0.5 (+ 6 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center 0 (* 0.5 (+ 7 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center (* 0.5 a2) (* 0.5 (+ 8 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center 0 (* 0.5 (+ 9 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     (make cylinder
				       (center (* 0.5 a2) (* 0.5 (+ 10 w) (sqrt 3)) 0) 
				       (radius rhole) 
				       (height hhole) 
				       (material hole-material))
				     ))
;add all holes to holes-list - will be added to geometry later
(define holes-list holes-list-first-row-a)
(set! holes-list (append holes-list holes-list-first-row-b))
(set! holes-list (append holes-list holes-list-second-row-a))
(set! holes-list (append holes-list holes-list-second-row-b))
(set! holes-list (append holes-list holes-list-third-row-a))
(set! holes-list (append holes-list holes-list-third-row-b))
(set! holes-list (append holes-list holes-list-remaining-rows-a))
(set! holes-list (append holes-list holes-list-remaining-rows-b))
;define lattice size and geometry end
;*****************************************************************

;defining variables and functions required for field integrations

;defining variables

(define new-field 0)
(define integral-value 0.0)
(define integral-value1 0.0)
(define integral-value2 0.0)
(define integral-value3 0.0)
(define point (vector3 0 0 0))
(define normal (vector3 0 0 0))
(define tangential (vector3 0 0 0))
(define outofplane (vector3 0 0 0))

;defining functions

(define (integral-out-of-plane-to-circle func x0 y0 z0 rad)
  ;path integral of func around a circle's edge in xy-plane z=z0 with thickness dz
  ;takes projection of field at point along unit vector the cylinder (i.e. out of plane of circle)
  ;set dz=1 for true path integral
  (begin
    (do ((theta 0 (+ theta dtheta))) ((>= theta (- 2pi tolerance)))
      (set! point (vector3 (+ x0 (* rad (cos theta))) (+ y0 (* rad (sin theta))) z0))
      (set! outofplane (vector3 0 0 1))
      (set! integral-value (+ integral-value
			      (* dz rad dtheta (func (get-field-point point) (get-epsilon-point point) outofplane))
			      )
	    )
      );end do
    );end begin
  );end define

(define (integral-out-of-plane-to-cylinder func cylinder)
  ;path integral of func around a cylinders edge for all z grid points within the slab
  (begin
    (do ((iz 0 (+ iz 1))) ((>= iz (length grid-points-list-in-slab-z)))
      (integral-out-of-plane-to-circle func 
			     (vector3-x (object-property-value cylinder 'center))
			     (vector3-y (object-property-value cylinder 'center))
			     (list-ref grid-points-list-in-slab-z iz)
			     (object-property-value cylinder 'radius))
      );end do
    );end begin
  );end define

(define (integral-out-of-plane-to-cylinder-list func cyl-list)
  ;path integral of function around the edge of all cylinders in list cyl-list
  (begin
    (do ((i 0 (+ i 1))) ((>= i (length cyl-list)))
      (integral-out-of-plane-to-cylinder func (list-ref cyl-list i))
      );end do
    );end begin
  );end define


(define (integral-normal-to-circle func x0 y0 z0 rad)
  ;path integral of func around a circle's edge in xy-plane z=z0 with thickness dz
  ;takes projection of field at point along unit vector normal to circle
  ;set dz=1 for true path integral
  (begin
    (do ((theta 0 (+ theta dtheta))) ((>= theta (- 2pi tolerance)))
      (set! point (vector3 (+ x0 (* rad (cos theta))) (+ y0 (* rad (sin theta))) z0))
      (set! normal (vector3 (cos theta) (sin theta) 0))
      (set! integral-value (+ integral-value
			      (* dz rad dtheta (func (get-field-point point) (get-epsilon-point point) normal))
			      )
	    )
      );end do
    );end begin
  );end define

(define (integral-normal-to-cylinder func cylinder)
  ;path integral of func around a cylinders edge for all z grid points within the slab
  (begin
    (do ((iz 0 (+ iz 1))) ((>= iz (length grid-points-list-in-slab-z)))
      (integral-normal-to-circle func 
			     (vector3-x (object-property-value cylinder 'center))
			     (vector3-y (object-property-value cylinder 'center))
			     (list-ref grid-points-list-in-slab-z iz)
			     (object-property-value cylinder 'radius))
      );end do
    );end begin
  );end define

(define (integral-normal-to-cylinder-list func cyl-list)
  ;path integral of function around the edge of all cylinders in list cyl-list
  (begin
    (do ((i 0 (+ i 1))) ((>= i (length cyl-list)))
      (integral-normal-to-cylinder func (list-ref cyl-list i))
      );end do
    );end begin
  );end define


(define (integral-tangential-to-circle func x0 y0 z0 rad)
  ;path integral of func around a circle's edge in xy-plane z=z0 with thickness dz
  ;takes projection of field at point along unit vector tangential to circle
  ;set dz=1 for true path integral
  (begin
    (do ((theta 0 (+ theta dtheta))) ((>= theta (- 2pi tolerance)))
      (set! point (vector3 (+ x0 (* rad (cos theta))) (+ y0 (* rad (sin theta))) z0))
      (set! tangential (vector3 (* -1 (sin theta)) (cos theta) 0))
      (set! integral-value (+ integral-value
			      (* dz rad dtheta (func (get-field-point point) (get-epsilon-point point) tangential))
			      )
	    )
      );end do
    );end begin
  );end define

(define (integral-tangential-to-cylinder func cylinder)
  ;path integral of func around a cylinders edge for all z grid points within the slab
  (begin
    (do ((iz 0 (+ iz 1))) ((>= iz (length grid-points-list-in-slab-z)))
      (integral-tangential-to-circle func 
			     (vector3-x (object-property-value cylinder 'center))
			     (vector3-y (object-property-value cylinder 'center))
			     (list-ref grid-points-list-in-slab-z iz)
			     (object-property-value cylinder 'radius))
      );end do
    );end begin
  );end define

(define (integral-tangential-to-cylinder-list func cyl-list)
  ;path integral of function around the edge of all cylinders in list cyl-list
  (begin
    (do ((i 0 (+ i 1))) ((>= i (length cyl-list)))
      (integral-tangential-to-cylinder func (list-ref cyl-list i))
      );end do
    );end begin
  );end define

;end of definitions

;*****************************************************************
;output grid-points begin
(define dx (/ (- xe xs) (vector3-x (get-grid-size))))
(define dy (/ (- ye ys) (vector3-y (get-grid-size))))
(define dz (/ (- ze zs) (vector3-z (get-grid-size))))
(define grid-points-list-x
  (list (* -0.5 (vector3-x (object-property-value geometry-lattice 'size))))
  )
(do ((i 1 (+ i 1))) ((>= i (vector3-x (get-grid-size))))
  (set! grid-points-list-x 
	(append grid-points-list-x 
		(list (+ (* i dx) (* -0.5 (vector3-x (object-property-value geometry-lattice 'size))))
		      )
		)
	)
  )
(define grid-points-list-y
  (list (* -0.5 (vector3-y (object-property-value geometry-lattice 'size))))
  )
(do ((i 1 (+ i 1))) ((>= i (vector3-y (get-grid-size))))
  (set! grid-points-list-y 
	(append grid-points-list-y 
		(list (+ (* i dy) (* -0.5 (vector3-y (object-property-value geometry-lattice 'size))))
		      )
		)
	)
  )
(define grid-points-list-z
  (list (* -0.5 (vector3-z (object-property-value geometry-lattice 'size))))
  )
(do ((i 1 (+ i 1))) ((>= i (vector3-z (get-grid-size))))
  (set! grid-points-list-z 
	(append grid-points-list-z 
		(list (+ (* i dz) (* -0.5 (vector3-z (object-property-value geometry-lattice 'size))))
		      )
		)
	)
  )
(define grid-points-list-in-slab-z (list ))
(do ((i 0 (+ i 1))) ((>= i (length grid-points-list-z)))
  (if (and (>= (list-ref grid-points-list-z i) (* -0.5 hslab))
	   (<= (list-ref grid-points-list-z i) (* 0.5 hslab)))
      (set! grid-points-list-in-slab-z 
	    (append grid-points-list-in-slab-z 
		    (list (list-ref grid-points-list-z i))
		    )
	    )
      );end if
  );end do
(print "x-grid-points: ")
(print (vector3-x (get-grid-size)) ": ")
(do ((i 0 (+ i 1))) ((>= i (length grid-points-list-x)))
  (print (list-ref grid-points-list-x i) " ")
  );end do
(print "\n")
(print "y-grid-points: ")
(print (vector3-y (get-grid-size)) ": ")
(do ((i 0 (+ i 1))) ((>= i (length grid-points-list-y)))
  (print (list-ref grid-points-list-y i) " ")
  );end do
(print "\n")
(print "z-grid-points: ")
(print (vector3-z (get-grid-size)) ": ")
(do ((i 0 (+ i 1))) ((>= i (length grid-points-list-z)))
  (print (list-ref grid-points-list-z i) " ")
  );end do
(print "\n")
(print "z-grid-points-in-slab: ")
(print (length grid-points-list-in-slab-z) ": ")
(do ((i 0 (+ i 1))) ((>= i (length grid-points-list-in-slab-z)))
  (print (list-ref grid-points-list-in-slab-z i) " ")
  );end do
(print "\n")
;output grid-points end
;*****************************************************************

;*****************************************************************
;define k-points begin
(define-param Ks 0.3)
(define-param Ke 0.5)
(define Kstart (vector3 Ks 0 0))
(define Kend (vector3 Ke 0 0))
(define-param Kinterp 19) ;number of k-points
(set! k-points (interpolate Kinterp (list Kstart Kend)))
;define k-points end
;*****************************************************************

;*****************************************************************
;define lattice geometry 
(set! geometry (list
		(make block  ;define slab with slab-material and size of the unit cell in x,y but hslab in z
		  (center 0 0 0) 
		  (material slab-material)
		  (size (vector3-x (object-property-value geometry-lattice 'size))
			(vector3-y (object-property-value geometry-lattice 'size))
			hslab))
		))
(set! geometry (append geometry holes-list))  ;add the list of holes
;define lattice geometry end
;*****************************************************************

(define pi (* 4 (atan 1)))
(define 2pi (* 8 (atan 1)))
(define dtheta (/ 2pi (vector3-y integration-resolution)))

;*****************************************************************
;define field functions 


(define (W1band-output-dfieldsquared b)
  (if (= b W1band)
      (fix-dfield-phase b)
	  (output-dpwr b)
      );end if
  );end define
(define (one F eps r)
  1
  );end define
(define (field-squared F eps r)
  (vector3-cdot F F)   ;F is complex - vector3-cdot does F*conj(F)
  );end define
(define (field-squared2 F) ;same as above, but without the redunant parameters passed
  (vector3-cdot F F)   ;F is complex - vector3-cdot does F*conj(F)
  );end define
(define (field-dot-unit-vector F eps unit-vector)
  (magnitude (vector3-dot F unit-vector)) ;give the projection of |field| along the unit-vector
  );end define
(define (field-dot-unit-vector-squared F eps unit-vector)
  (expt (magnitude (vector3-dot F unit-vector)) 2) ;give the projection of |field|^2 along the unit-vector
  );end define
(define (field-dot-unit-vector-cmplx F eps unit-vector)
  (vector3-dot F unit-vector) ;give the projection of field along the unit-vector
  );end define
(define (field-dot-unit-vector-squared-cmplx F eps unit-vector)
  (expt (vector3-dot F unit-vector) 2) ;give the projection of field^2 along the unit-vector
  );end define

;define field functions end
;*****************************************************************


;*****************************************************************
;define integral functions and variables

(define integral-value 0.0)
(define integral-value1 0.0)
(define integral-value2 0.0)
(define integral-value3 0.0)
(define integral-value4 0.0)
(define integral-value5 0.0)
(define integral-value6 0.0)

;calculation of backscattering rho coeeficient
(define (integral-rho-holes-list-first-row b)
  ;does the surface integral of rho = |(Etangentail)^2 + (Dnormal)^2/(eps1*eps2)|^2 for the first row
  ;the first row consists of two holes, saved in holes-list-first-row-a and holes-list-first-row-b
  ;do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently
  (begin
    (print "Calling integral-rho-holes-list-first-row...\n")
    (print "does the surface integral of rho = (Etangentail)^2 + (Dnormal)^2/(eps1*eps2) on cylinders in first row\n")
    (print "calculates |int[alpha2]|^2\n")
    (print "the first row consists of two holes, saved in holes-list-first-row-a and holes-list-first-row-b\n")
    (print "do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently\n")
    (print "holes-list-first-row-a contains " (length holes-list-first-row-a) " cylinders \n")
    (print "holes-list-first-row-b contains " (length holes-list-first-row-b) " cylinders \n")
    (set! integral-value 0.0)
    (set! integral-value1 0.0)
    (set! integral-value2 0.0)
    (set! integral-value3 0.0)
    (set! integral-value4 0.0)
    (print "Resetting integral-value to zero...\n")
    (print "Calculating integral-value for z grid points in the slab...\n") 
    (print "...using grid parameters: dtheta=" dtheta " (" (vector3-y integration-resolution)  
	   ") dz=" dz " (" (length grid-points-list-in-slab-z)
	   ") \n") 
    (get-efield b)
    (print "Getting efield for band " b "\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-first-row-a)	;integral of Etangential^2
    (print "Integrating tangential component of efield on hole edges for holes-list-first-row-a... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value1 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-first-row-b)	;integral of Etangential^2
    (print "Integrating tangential component of efield on hole edges for holes-list-first-row-b... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value3 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (get-dfield b)
    (print "Getting dfield for band " b "\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-first-row-a) 	;integral of Dnormal^2
    (print "Integrating normal component of dfield on hole edges for holes-list-first-row-a... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value2 
	  (/ integral-value 
	   (* (object-property-value slab-material 'epsilon) 
	      (object-property-value hole-material 'epsilon)))) 
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-first-row-b) 	;integral of Dnormal^2
    (print "Integrating normal component of dfield on hole edges for holes-list-first-row-b... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value4 
	  (/ integral-value 
	   (* (object-property-value slab-material 'epsilon) 
	      (object-property-value hole-material 'epsilon)))) 
    (print "Using epsilon values " (object-property-value slab-material 'epsilon) " and "
	   (object-property-value hole-material 'epsilon) "\n")
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (set! integral-value (+ (expt (magnitude (+ integral-value1 integral-value2)) 2) (expt (magnitude (+ integral-value3 integral-value4)) 2)))		;calculating the modulus squared of the integral on each hole, and adding contributions incoherently
    (print "Taking square of integral-value\n")
    (print "integral-rho-holes-list-first-row:, Band, " b ", k-point, " 
	   (vector3-x current-k)", "
	   (vector3-y current-k)", "
	   (vector3-z current-k)", "
	   "tangential-efield-hole-a, " integral-value1 ", "
	   "tangential-efield-hole-b, " integral-value3 ", "
	   "normal-dfield-hole-a, " integral-value2 ", "
	   "normal-dfield-hole-b, " integral-value4 ", "
	   "integral-value, " integral-value
	   "\n")
    );end begin
  );end define

(define (integral-rho-holes-list-second-row b)
  ;band function to pass to run
  ;does the surface integral of rho = |(Etangentail)^2 + (Dnormal)^2/(eps1*eps2)|^2 for the second row
  ;the second row consists of two holes, saved in holes-list-second-row-a and holes-list-second-row-b
  ;do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently
  (begin
    (print "Calling integral-rho-holes-list-second-row...\n")
    (print "does the surface integral of rho = (Etangentail)^2 + (Dnormal)^2/(eps1*eps2) on cylinders in second row\n")
    (print "calculates |int[alpha2]|^2\n")
    (print "the second row consists of two holes, saved in holes-list-second-row-a and holes-list-second-row-b\n")
    (print "do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently\n")
    (print "holes-list-second-row-a contains " (length holes-list-second-row-a) " cylinders \n")
    (print "holes-list-second-row-b contains " (length holes-list-second-row-b) " cylinders \n")
    (set! integral-value 0.0)
    (set! integral-value1 0.0)
    (set! integral-value2 0.0)
    (set! integral-value3 0.0)
    (set! integral-value4 0.0)
    (print "Resetting integral-value to zero...\n")
    (print "Calculating integral-value for z grid points in the slab...\n") 
    (print "...using grid parameters: dtheta=" dtheta " (" (vector3-y integration-resolution)  
	   ") dz=" dz " (" (length grid-points-list-in-slab-z)
	   ") \n") 
    (get-efield b)
    (print "Getting efield for band " b "\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-second-row-a)	;integral of Etangential^2
    (print "Integrating tangential component of efield on hole edges for holes-list-second-row-a... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value1 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-second-row-b)	;integral of Etangential^2
    (print "Integrating tangential component of efield on hole edges for holes-list-second-row-b... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value3 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (get-dfield b)
    (print "Getting dfield for band " b "\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-second-row-a) 	;integral of Dnormal^2
    (print "Integrating normal component of dfield on hole edges for holes-list-second-row-a... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value2 
	  (/ integral-value 
	   (* (object-property-value slab-material 'epsilon) 
	      (object-property-value hole-material 'epsilon)))) 
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-second-row-b) 	;integral of Dnormal^2
    (print "Integrating normal component of dfield on hole edges for holes-list-second-row-b... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value4 
	  (/ integral-value 
	   (* (object-property-value slab-material 'epsilon) 
	      (object-property-value hole-material 'epsilon)))) 
    (print "Using epsilon values " (object-property-value slab-material 'epsilon) " and "
	   (object-property-value hole-material 'epsilon) "\n")
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (set! integral-value (+ (expt (magnitude (+ integral-value1 integral-value2)) 2) (expt (magnitude (+ integral-value3 integral-value4)) 2)))		;calculating the modulus squared of the integral on each hole, and adding contributions incoherently
    (print "Taking square of integral-value\n")
    (print "integral-rho-holes-list-second-row:, Band, " b ", k-point, " 
	   (vector3-x current-k)", "
	   (vector3-y current-k)", "
	   (vector3-z current-k)", "
	   "tangential-efield-hole-a, " integral-value1 ", "
	   "tangential-efield-hole-b, " integral-value3 ", "
	   "normal-dfield-hole-a, " integral-value2 ", "
	   "normal-dfield-hole-b, " integral-value4 ", "
	   "integral-value, " integral-value
	   "\n")
    );end begin
  );end define










(define (integral-rho-holes-list-third-row b)
  ;does the surface integral of rho = |(Etangentail)^2 + (Dnormal)^2/(eps1*eps2)|^2 for the third row
  ;the third row consists of two holes, saved in holes-list-third-row-a and holes-list-third-row-b
  ;do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently
  (begin
    (print "Calling integral-rho-holes-list-third-row...\n")
    (print "does the surface integral of rho = (Etangentail)^2 + (Dnormal)^2/(eps1*eps2) on cylinders in third row\n")
    (print "calculates |int[alpha2]|^2\n")
    (print "the third row consists of two holes, saved in holes-list-third-row-a and holes-list-third-row-b\n")
    (print "do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently\n")
    (print "holes-list-third-row-a contains " (length holes-list-third-row-a) " cylinders \n")
    (print "holes-list-third-row-b contains " (length holes-list-third-row-b) " cylinders \n")
    (set! integral-value 0.0)
    (set! integral-value1 0.0)
    (set! integral-value2 0.0)
    (set! integral-value3 0.0)
    (set! integral-value4 0.0)
    (print "Resetting integral-value to zero...\n")
    (print "Calculating integral-value for z grid points in the slab...\n") 
    (print "...using grid parameters: dtheta=" dtheta " (" (vector3-y integration-resolution)  
	   ") dz=" dz " (" (length grid-points-list-in-slab-z)
	   ") \n") 
    (get-efield b)
    (print "Getting efield for band " b "\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-third-row-a)	;integral of Etangential^2
    (print "Integrating tangential component of efield on hole edges for holes-list-third-row-a... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value1 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-third-row-b)	;integral of Etangential^2
    (print "Integrating tangential component of efield on hole edges for holes-list-third-row-b... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value3 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (get-dfield b)
    (print "Getting dfield for band " b "\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-third-row-a) 	;integral of Dnormal^2
    (print "Integrating normal component of dfield on hole edges for holes-list-third-row-a... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value2 
	  (/ integral-value 
	   (* (object-property-value slab-material 'epsilon) 
	      (object-property-value hole-material 'epsilon)))) 
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-squared-cmplx holes-list-third-row-b) 	;integral of Dnormal^2
    (print "Integrating normal component of dfield on hole edges for holes-list-third-row-b... \n")
    (print "     ...answer is " integral-value  " \n")
    (set! integral-value4 
	  (/ integral-value 
	   (* (object-property-value slab-material 'epsilon) 
	      (object-property-value hole-material 'epsilon)))) 
    (print "Using epsilon values " (object-property-value slab-material 'epsilon) " and "
	   (object-property-value hole-material 'epsilon) "\n")
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (set! integral-value (+ (expt (magnitude (+ integral-value1 integral-value2)) 2) (expt (magnitude (+ integral-value3 integral-value4)) 2)))		;calculating the modulus squared of the integral on each hole, and adding contributions incoherently
    (print "Taking square of integral-value\n")
    (print "integral-rho-holes-list-third-row:, Band, " b ", k-point, " 
	   (vector3-x current-k)", "
	   (vector3-y current-k)", "
	   (vector3-z current-k)", "
	   "tangential-efield-hole-a, " integral-value1 ", "
	   "tangential-efield-hole-b, " integral-value3 ", "
	   "normal-dfield-hole-a, " integral-value2 ", "
	   "normal-dfield-hole-b, " integral-value4 ", "
	   "integral-value, " integral-value
	   "\n")
    );end begin
  );end define




;Calculation of out of plane coefficient gamma
(define (integral-gamma-holes-list-first-row b)
  ;Out of plane solution for first row
  ;the first row consists of two holes, saved in holes-list-first-row-a and holes-list-first-row-b
  ;do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently
  (begin
    (print "Calling integral-gamma-holes-list-first-row...\n")
    (print "...does |int[Etan]|^2+|int[Eout]|^2+|int[Dnorm]|^2/ep1"
           "      + |int[Etan]|^2+|int[Eout]|^2+|int[Dnorm]|^2/ep2\n")
    (print "the first row consists of two holes, saved in holes-list-first-row-a and holes-list-first-row-b\n")
    (print "do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently\n")
    (print "holes-list-first-row-a contains " (length holes-list-first-row-a) " cylinders \n")
    (print "holes-list-first-row-b contains " (length holes-list-first-row-b) " cylinders \n")
    (set! integral-value 0.0)
    (set! integral-value1 0.0);E tangential
    (set! integral-value2 0.0);E out-of-plane
    (set! integral-value3 0.0);D normal
    (set! integral-value4 0.0);E tangential
    (set! integral-value5 0.0);E out-of-plane
    (set! integral-value6 0.0);D normal
    (print "Resetting integral-value to zero...\n")
    (print "Calculating integral-value for z grid points in the slab...\n") 
    (print "...using grid parameters: dtheta=" dtheta " (" 
	   (vector3-y integration-resolution)  
	   ") dz=" dz " (" (length grid-points-list-in-slab-z)
	   ") \n") 
    (get-efield b)
    (print "Getting efield for band " b "\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-cmplx
					  holes-list-first-row-a)
    (print "Integrating tangential component of efield on hole edges for first-row-a... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value1 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-cmplx
					  holes-list-first-row-b)
    (print "Integrating tangential component of efield on hole edges for first-row-b... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value4 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-out-of-plane-to-cylinder-list field-dot-unit-vector-cmplx 
					    holes-list-first-row-a)
    (print "Integrating out-of-plane component of efield on hole edges for first-row-a... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value2 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-out-of-plane-to-cylinder-list field-dot-unit-vector-cmplx 
					    holes-list-first-row-b)
    (print "Integrating out-of-plane component of efield on hole edges for first-row-b... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value5 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (get-dfield b)
    (print "Getting dfield for band " b "\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-cmplx 
				      holes-list-first-row-a)
    (print "Integrating normal component of dfield on hole edges for first-row-a... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value3 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-cmplx 
				      holes-list-first-row-b)
    (print "Integrating normal component of dfield on hole edges for first-row-b... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value6 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (print "Using epsilon values " (object-property-value slab-material 'epsilon) " and "
	(object-property-value hole-material 'epsilon) "\n")
    (set! integral-value
	  (+ (* integral-value1 (conj integral-value1))
	     (* integral-value2 (conj integral-value2))
	     (* (/ integral-value3 
		   (object-property-value slab-material 'epsilon))
		(conj (/ integral-value3 
			 (object-property-value slab-material 'epsilon))))
	     (* integral-value1 (conj integral-value1))
	     (* integral-value2 (conj integral-value2))
	     (* (/ integral-value3 
		   (object-property-value hole-material 'epsilon))
		(conj (/ integral-value3 
			 (object-property-value hole-material 'epsilon))))

	     (* integral-value4 (conj integral-value4))
	     (* integral-value5 (conj integral-value5))
	     (* (/ integral-value6 
		   (object-property-value slab-material 'epsilon))
		(conj (/ integral-value6 
			 (object-property-value slab-material 'epsilon))))
	     (* integral-value4 (conj integral-value4))
	     (* integral-value5 (conj integral-value5))
	     (* (/ integral-value6 
		   (object-property-value hole-material 'epsilon))
		(conj (/ integral-value6 
			 (object-property-value hole-material 'epsilon))))
	     )
	  )
    (print "integral-gamma-holes-list-first-row:, Band, " b ", k-point, " 
	   (vector3-x current-k)", "
	   (vector3-y current-k)", "
	   (vector3-z current-k)", "
	   "tangential-efield-hole-a, |int[Etan]|^2, " integral-value1 ", "
	   "tangential-efield-hole-b, |int[Etan]|^2, " integral-value4 ", "
	   "outofplane-efield-hole-a, |int[Eout]|^2, " integral-value2 ", "
	   "outofplane-efield-hole-b, |int[Eout]|^2, " integral-value5 ", "
	   "normal-dfield-hole-a, |int[Dnorm]|^2, " integral-value3 ", "
	   "normal-dfield-hole-b, |int[Dnorm]|^2, " integral-value6 ", "
	   "integral-value, " integral-value
	   "\n")
    );end begin
  );end define

(define (integral-gamma-holes-list-second-row b)
  ;Out of plane solution for second row
  ;the second row consists of two holes, saved in holes-list-second-row-a and holes-list-second-row-b
  ;do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently
  (begin
    (print "Calling integral-gamma-holes-list-second-row...\n")
    (print "...does |int[Etan]|^2+|int[Eout]|^2+|int[Dnorm]|^2/ep1"
           "      + |int[Etan]|^2+|int[Eout]|^2+|int[Dnorm]|^2/ep2\n")
    (print "the second row consists of two holes, saved in holes-list-second-row-a and holes-list-second-row-b\n")
    (print "do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently\n")
    (print "holes-list-second-row-a contains " (length holes-list-second-row-a) " cylinders \n")
    (print "holes-list-second-row-b contains " (length holes-list-second-row-b) " cylinders \n")
    (set! integral-value 0.0)
    (set! integral-value1 0.0);E tangential
    (set! integral-value2 0.0);E out-of-plane
    (set! integral-value3 0.0);D normal
    (set! integral-value4 0.0);E tangential
    (set! integral-value5 0.0);E out-of-plane
    (set! integral-value6 0.0);D normal
    (print "Resetting integral-value to zero...\n")
    (print "Calculating integral-value for z grid points in the slab...\n") 
    (print "...using grid parameters: dtheta=" dtheta " (" 
	   (vector3-y integration-resolution)  
	   ") dz=" dz " (" (length grid-points-list-in-slab-z)
	   ") \n") 
    (get-efield b)
    (print "Getting efield for band " b "\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-cmplx
					  holes-list-second-row-a)
    (print "Integrating tangential component of efield on hole edges for second-row-a... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value1 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-cmplx
					  holes-list-second-row-b)
    (print "Integrating tangential component of efield on hole edges for second-row-b... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value4 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-out-of-plane-to-cylinder-list field-dot-unit-vector-cmplx 
					    holes-list-second-row-a)
    (print "Integrating out-of-plane component of efield on hole edges for second-row-a... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value2 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-out-of-plane-to-cylinder-list field-dot-unit-vector-cmplx 
					    holes-list-second-row-b)
    (print "Integrating out-of-plane component of efield on hole edges for second-row-b... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value5 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (get-dfield b)
    (print "Getting dfield for band " b "\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-cmplx 
				      holes-list-second-row-a)
    (print "Integrating normal component of dfield on hole edges for second-row-a... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value3 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-cmplx 
				      holes-list-second-row-b)
    (print "Integrating normal component of dfield on hole edges for second-row-b... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value6 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (print "Using epsilon values " (object-property-value slab-material 'epsilon) " and "
	(object-property-value hole-material 'epsilon) "\n")
    (set! integral-value
	  (+ (* integral-value1 (conj integral-value1))
	     (* integral-value2 (conj integral-value2))
	     (* (/ integral-value3 
		   (object-property-value slab-material 'epsilon))
		(conj (/ integral-value3 
			 (object-property-value slab-material 'epsilon))))
	     (* integral-value1 (conj integral-value1))
	     (* integral-value2 (conj integral-value2))
	     (* (/ integral-value3 
		   (object-property-value hole-material 'epsilon))
		(conj (/ integral-value3 
			 (object-property-value hole-material 'epsilon))))

	     (* integral-value4 (conj integral-value4))
	     (* integral-value5 (conj integral-value5))
	     (* (/ integral-value6 
		   (object-property-value slab-material 'epsilon))
		(conj (/ integral-value6 
			 (object-property-value slab-material 'epsilon))))
	     (* integral-value4 (conj integral-value4))
	     (* integral-value5 (conj integral-value5))
	     (* (/ integral-value6 
		   (object-property-value hole-material 'epsilon))
		(conj (/ integral-value6 
			 (object-property-value hole-material 'epsilon))))
	     )
	  )
    (print "integral-gamma-holes-list-second-row:, Band, " b ", k-point, " 
	   (vector3-x current-k)", "
	   (vector3-y current-k)", "
	   (vector3-z current-k)", "
	   "tangential-efield-hole-a, |int[Etan]|^2, " integral-value1 ", "
	   "tangential-efield-hole-b, |int[Etan]|^2, " integral-value4 ", "
	   "outofplane-efield-hole-a, |int[Eout]|^2, " integral-value2 ", "
	   "outofplane-efield-hole-b, |int[Eout]|^2, " integral-value5 ", "
	   "normal-dfield-hole-a, |int[Dnorm]|^2, " integral-value3 ", "
	   "normal-dfield-hole-b, |int[Dnorm]|^2, " integral-value6 ", "
	   "integral-value, " integral-value
	   "\n")
    );end begin
  );end define


(define (integral-gamma-holes-list-third-row b)
  ;Out of plane solution for third row
  ;the third row consists of two holes, saved in holes-list-third-row-a and holes-list-third-row-b
  ;do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently
  (begin
    (print "Calling integral-gamma-holes-list-third-row...\n")
    (print "...does |int[Etan]|^2+|int[Eout]|^2+|int[Dnorm]|^2/ep1"
           "      + |int[Etan]|^2+|int[Eout]|^2+|int[Dnorm]|^2/ep2\n")
    (print "the third row consists of two holes, saved in holes-list-third-row-a and holes-list-third-row-b\n")
    (print "do the integrals on each hole taking spatial phase into account, but add the two contributions incoherently\n")
    (print "holes-list-third-row-a contains " (length holes-list-third-row-a) " cylinders \n")
    (print "holes-list-third-row-b contains " (length holes-list-third-row-b) " cylinders \n")
    (set! integral-value 0.0)
    (set! integral-value1 0.0);E tangential
    (set! integral-value2 0.0);E out-of-plane
    (set! integral-value3 0.0);D normal
    (set! integral-value4 0.0);E tangential
    (set! integral-value5 0.0);E out-of-plane
    (set! integral-value6 0.0);D normal
    (print "Resetting integral-value to zero...\n")
    (print "Calculating integral-value for z grid points in the slab...\n") 
    (print "...using grid parameters: dtheta=" dtheta " (" 
	   (vector3-y integration-resolution)  
	   ") dz=" dz " (" (length grid-points-list-in-slab-z)
	   ") \n") 
    (get-efield b)
    (print "Getting efield for band " b "\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-cmplx
					  holes-list-third-row-a)
    (print "Integrating tangential component of efield on hole edges for third-row-a... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value1 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-tangential-to-cylinder-list field-dot-unit-vector-cmplx
					  holes-list-third-row-b)
    (print "Integrating tangential component of efield on hole edges for third-row-b... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value4 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-out-of-plane-to-cylinder-list field-dot-unit-vector-cmplx 
					    holes-list-third-row-a)
    (print "Integrating out-of-plane component of efield on hole edges for third-row-a... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value2 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-out-of-plane-to-cylinder-list field-dot-unit-vector-cmplx 
					    holes-list-third-row-b)
    (print "Integrating out-of-plane component of efield on hole edges for third-row-b... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value5 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (get-dfield b)
    (print "Getting dfield for band " b "\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-cmplx 
				      holes-list-third-row-a)
    (print "Integrating normal component of dfield on hole edges for third-row-a... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value3 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (integral-normal-to-cylinder-list field-dot-unit-vector-cmplx 
				      holes-list-third-row-b)
    (print "Integrating normal component of dfield on hole edges for third-row-b... \n")
    (print "    ...answer is " integral-value "\n")
    (set! integral-value6 integral-value)
    (set! integral-value 0.0)
    (print "Resetting integral-value to zero...\n")
    (print "Using epsilon values " (object-property-value slab-material 'epsilon) " and "
	(object-property-value hole-material 'epsilon) "\n")
    (set! integral-value
	  (+ (* integral-value1 (conj integral-value1))
	     (* integral-value2 (conj integral-value2))
	     (* (/ integral-value3 
		   (object-property-value slab-material 'epsilon))
		(conj (/ integral-value3 
			 (object-property-value slab-material 'epsilon))))
	     (* integral-value1 (conj integral-value1))
	     (* integral-value2 (conj integral-value2))
	     (* (/ integral-value3 
		   (object-property-value hole-material 'epsilon))
		(conj (/ integral-value3 
			 (object-property-value hole-material 'epsilon))))

	     (* integral-value4 (conj integral-value4))
	     (* integral-value5 (conj integral-value5))
	     (* (/ integral-value6 
		   (object-property-value slab-material 'epsilon))
		(conj (/ integral-value6 
			 (object-property-value slab-material 'epsilon))))
	     (* integral-value4 (conj integral-value4))
	     (* integral-value5 (conj integral-value5))
	     (* (/ integral-value6 
		   (object-property-value hole-material 'epsilon))
		(conj (/ integral-value6 
			 (object-property-value hole-material 'epsilon))))
	     )
	  )
    (print "integral-gamma-holes-list-third-row:, Band, " b ", k-point, " 
	   (vector3-x current-k)", "
	   (vector3-y current-k)", "
	   (vector3-z current-k)", "
	   "tangential-efield-hole-a, |int[Etan]|^2, " integral-value1 ", "
	   "tangential-efield-hole-b, |int[Etan]|^2, " integral-value4 ", "
	   "outofplane-efield-hole-a, |int[Eout]|^2, " integral-value2 ", "
	   "outofplane-efield-hole-b, |int[Eout]|^2, " integral-value5 ", "
	   "normal-dfield-hole-a, |int[Dnorm]|^2, " integral-value3 ", "
	   "normal-dfield-hole-b, |int[Dnorm]|^2, " integral-value6 ", "
	   "integral-value, " integral-value
	   "\n")
    );end begin
  );end define

;define integral functions end
;*****************************************************************



;*****************************************************************
;band functions to pass to run
(define-param W1band 23)  ;define the band number of intrest
;if you change the total number of holes, then this number may alter
(define (W1band-output-efield b)
   (if (= b W1band)
       (begin 
	 (fix-efield-phase b)
	 (output-efield b)
	 );end begin
       );end if
   );end define
(define (W1band-output-hfield b)
   (if (= b W1band)
       (begin 
	 (fix-hfield-phase b)
	 (output-hfield b)
	 );end begin
       );end if
   );end define
(define (W1band-integral-gamma-holes-list-first-row b)
  ;band function to pass to run, calls the calculation of out of plane coefficient on first row
  (if (= b W1band)
      (integral-gamma-holes-list-first-row b)
      );end if
  );end define
(define (W1band-integral-gamma-holes-list-second-row b)
  ;band function to pass to run, calls the calculation of out of plane coefficient on second row
  (if (= b W1band)
      (integral-gamma-holes-list-second-row b)
      );end if
  );end define
(define (W1band-integral-gamma-holes-list-third-row b)
  ;band function to pass to run, calls the calculation of out of plane coefficient on third row
  (if (= b W1band)
      (integral-gamma-holes-list-third-row b)
      );end if
  );end define
(define (W1band-integral-rho-holes-list-first-row b)
  ;band function to pass to run
  ;calls calculation of rho for the first row
  (if (= b W1band)
      (integral-rho-holes-list-first-row b)
      );end if
  );end define
(define (W1band-integral-rho-holes-list-second-row b)
  ;band function to pass to run
  ;calls calculation of rho for the second row
  (if (= b W1band)
      (integral-rho-holes-list-second-row b)
      );end if
  );end define
(define (W1band-integral-rho-holes-list-third-row b)
  ;band function to pass to run
  ;calls calculation of rho for the third row
  (if (= b W1band)
      (integral-rho-holes-list-third-row b)
      );end if
  );end define
;band functions to pass to run end
;*****************************************************************




  
;*****************************************************************
;run calculations begin
(if (= calculation-type 0);basic calculation of bandstructures
    (run-zeven))
(if (= calculation-type 1);bandstructures plus group velocities
    (run-zeven display-group-velocities))
(if (= calculation-type 2);bandstructures plus output fields profiles
    (run-zeven W1band-output-efield
	       W1band-output-hfield
		 W1band-output-dfieldsquared))
(if (= calculation-type 3);bandstructures plus group velocities plus output fields profiles
    (run-zeven display-group-velocities 
	       W1band-output-efield
	       W1band-output-hfield
		 W1band-output-dfieldsquared))
(if (= calculation-type 4);bandstructures plus group velocities plus calculate integrals
    (run-zeven display-group-velocities 
	       W1band-integral-rho-holes-list-first-row
	       W1band-integral-rho-holes-list-second-row
	       W1band-integral-rho-holes-list-third-row
	       W1band-integral-gamma-holes-list-first-row
	       W1band-integral-gamma-holes-list-second-row
	       W1band-integral-gamma-holes-list-third-row))
(if (= calculation-type 5);bandstructures plus group velocities plus calculate integrals plus output fields profiles
    (run-zeven display-group-velocities 
	       W1band-integral-rho-holes-list-first-row
	       W1band-integral-rho-holes-list-second-row
	       W1band-integral-rho-holes-list-third-row
	       W1band-integral-gamma-holes-list-first-row
	       W1band-integral-gamma-holes-list-second-row
	       W1band-integral-gamma-holes-list-third-row
	       W1band-output-efield
	       W1band-output-hfield
		 W1band-output-dfieldsquared))
(if (= calculation-type 6); For Neural networks summer project 2022 
    (run-zeven display-yparities))
(if (= calculation-type 9);for testing
    (begin 
      )
    )
;run calculations end
;*****************************************************************


;(set-param! interactive? true)

