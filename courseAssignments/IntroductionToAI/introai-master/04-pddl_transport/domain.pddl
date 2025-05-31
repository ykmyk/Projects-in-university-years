(define (domain transport)
  (:predicates
    (car ?c)
    (box ?b)
    (place ?p)
    (at ?x ?p)       ; object (car or box) x is at place p
    (in ?b ?c)        ; box b is inside car c
    (empty ?c)        ; car c is empty
  )

  (:action move
    :parameters (?c ?from ?to)
    :precondition (and
      (car ?c)
      (place ?from)
      (place ?to)
      (at ?c ?from)
    )
    :effect (and
      (not (at ?c ?from))
      (at ?c ?to)
    )
  )

  (:action load
    :parameters (?b ?c ?p)
    :precondition (and
      (box ?b)
      (car ?c)
      (place ?p)
      (at ?b ?p)
      (at ?c ?p)
      (empty ?c)
    )
    :effect (and
      (not (at ?b ?p))
      (not (empty ?c))
      (in ?b ?c)
      (at ?b ?c)  ; so goal (at box1 car1) can be matched
    )
  )

  (:action unload
    :parameters (?b ?c ?p)
    :precondition (and
      (car ?c)
      (box ?b)
      (place ?p)
      (at ?c ?p)
      (at ?b ?c) ; this is now the main way we know the box is in the car
    )
    :effect (and
      (not (at ?b ?c)) ; remove the fact that the box is inside the car
      (not (in ?b ?c)) ; cleanup for robustness
      (at ?b ?p)
      (empty ?c)
    )
  )

)